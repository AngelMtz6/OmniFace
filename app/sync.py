"""
OmniFace — sincronización Git de la base de datos de identidades.

Flujo:
  Web (registro nuevo) → push_db()          → git add DB + commit + push
  Desktop (botón)      → pull_db_identities() → git fetch → leer DB remota
                          → importar solo identidades/muestras nuevas al DB local
                          → NO toca access_log, cameras ni accounts locales
"""
import os
import sqlite3
import subprocess
import tempfile
import threading

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DB   = os.path.join(_ROOT, "data", "omniface.db")

# Mutex: evita que dos operaciones git corran simultáneamente
_GIT_LOCK = threading.Lock()


# ── PUSH (llamado desde el servidor web) ─────────────────────────────────────

def push_db(nombre: str = "") -> None:
    """
    Hace commit + push de data/omniface.db en un hilo de fondo.
    No bloquea la respuesta HTTP — se lanza y se olvida.
    """
    label = f"{nombre} — " if nombre else ""

    def _run():
        with _GIT_LOCK:
            try:
                msg = f"sync: {label}nuevo registro facial via web"

                r_add = subprocess.run(
                    ["git", "add", "data/omniface.db"],
                    cwd=_ROOT, capture_output=True, text=True, timeout=15
                )
                if r_add.returncode != 0:
                    print(f"[OmniFace Sync] git add falló: {r_add.stderr.strip()}")
                    return

                r_commit = subprocess.run(
                    ["git", "commit", "-m", msg, "--no-verify"],
                    cwd=_ROOT, capture_output=True, text=True, timeout=15
                )
                # "nothing to commit" no es un error real
                if "nothing to commit" in (r_commit.stdout + r_commit.stderr):
                    print("[OmniFace Sync] DB sin cambios, nada que pushear")
                    return
                if r_commit.returncode != 0:
                    print(f"[OmniFace Sync] git commit falló: {r_commit.stderr.strip()}")
                    return

                r_push = subprocess.run(
                    ["git", "push", "origin", "main"],
                    cwd=_ROOT, capture_output=True, text=True, timeout=30
                )
                if r_push.returncode == 0:
                    print(f"[OmniFace Sync] ✓ Sincronizado: {msg}")
                else:
                    # Push rechazado (otro push llegó primero) — pull --rebase y reintento
                    print("[OmniFace Sync] Push rechazado, intentando rebase…")
                    subprocess.run(
                        ["git", "pull", "--rebase", "origin", "main"],
                        cwd=_ROOT, capture_output=True, timeout=20
                    )
                    r2 = subprocess.run(
                        ["git", "push", "origin", "main"],
                        cwd=_ROOT, capture_output=True, text=True, timeout=30
                    )
                    if r2.returncode == 0:
                        print(f"[OmniFace Sync] ✓ Sincronizado (reintento): {msg}")
                    else:
                        print(f"[OmniFace Sync] Push falló: {r2.stderr.strip()}")

            except subprocess.TimeoutExpired:
                print("[OmniFace Sync] Timeout en operación git")
            except Exception as e:
                print(f"[OmniFace Sync] Error inesperado: {e}")

    threading.Thread(target=_run, daemon=True).start()


# ── PULL (llamado desde la app de escritorio) ─────────────────────────────────

def pull_db_identities() -> tuple[bool, str]:
    """
    Trae el DB remoto como archivo temporal y SOLO importa las identidades
    (+ muestras faciales) que no existen en el DB local.

    No reemplaza el DB completo → los access_log, cameras y accounts locales
    se conservan intactos.

    Retorna (éxito: bool, mensaje: str).
    """
    with _GIT_LOCK:
        tmp_path = None
        try:
            # 1. Fetch — actualiza refs remotas sin tocar el working tree
            r_fetch = subprocess.run(
                ["git", "fetch", "origin", "main"],
                cwd=_ROOT, capture_output=True, text=True, timeout=25
            )
            if r_fetch.returncode != 0:
                return False, f"git fetch falló: {r_fetch.stderr.strip()}"

            # 2. Leer el DB remoto como bytes (NO text=True — es binario)
            r_show = subprocess.run(
                ["git", "show", "origin/main:data/omniface.db"],
                cwd=_ROOT, capture_output=True, timeout=15
            )
            if r_show.returncode != 0:
                return False, "No se pudo leer data/omniface.db del remoto"

            # 3. Escribir a archivo temporal
            with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
                tmp.write(r_show.stdout)
                tmp_path = tmp.name

            # 4. Abrir ambas conexiones
            remote = sqlite3.connect(tmp_path)
            remote.row_factory = sqlite3.Row
            local  = sqlite3.connect(_DB, timeout=20)
            local.execute("PRAGMA foreign_keys = ON")

            new_identities = 0
            new_samples    = 0

            try:
                # Identidades del remoto que tengan account_id (registradas via web)
                # o cualquiera que no exista localmente
                remote_ids = remote.execute(
                    "SELECT * FROM identities ORDER BY created_at"
                ).fetchall()

                for row in remote_ids:
                    iid = row["id"]

                    # ¿Ya existe localmente?
                    exists = local.execute(
                        "SELECT id FROM identities WHERE id = ?", (iid,)
                    ).fetchone()
                    if exists:
                        continue  # ya la tenemos

                    # Insertar identidad
                    cols = [desc[0] for desc in remote.execute(
                        "SELECT * FROM identities WHERE id = ?", (iid,)
                    ).description]
                    vals = tuple(row[c] for c in cols)
                    placeholders = ", ".join(["?"] * len(cols))
                    local.execute(
                        f"INSERT OR IGNORE INTO identities ({', '.join(cols)}) "
                        f"VALUES ({placeholders})",
                        vals
                    )
                    new_identities += 1

                    # Insertar muestras faciales de esa identidad
                    samples = remote.execute(
                        "SELECT * FROM face_samples WHERE identity_id = ?", (iid,)
                    ).fetchall()
                    for s in samples:
                        local.execute(
                            "INSERT OR IGNORE INTO face_samples (id, identity_id, face_data) "
                            "VALUES (?, ?, ?)",
                            (s["id"], s["identity_id"], bytes(s["face_data"]))
                        )
                        new_samples += 1

                local.commit()

                if new_identities == 0:
                    return True, "Sin cambios — ya tienes todas las identidades"

                return True, (
                    f"{new_identities} identidad(es) nueva(s) importada(s) "
                    f"({new_samples} muestras faciales)"
                )

            finally:
                remote.close()
                local.close()

        except subprocess.TimeoutExpired:
            return False, "Timeout al contactar el servidor Git"
        except Exception as e:
            return False, f"Error inesperado: {e}"
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
