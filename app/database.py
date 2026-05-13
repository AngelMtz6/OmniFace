import sqlite3
import os
import threading
import subprocess
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'omniface.db')


def get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=20)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def git_sync():
    """Ejecuta un push a Git en segundo plano para sincronizar la DB."""
    def _sync():
        try:
            # 1. Checkpoint para asegurar que todo el WAL esté en el .db principal
            conn = get_connection()
            conn.execute("PRAGMA wal_checkpoint(FULL)")
            conn.close()

            # Directorio raíz del proyecto
            root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            
            # 2. Añadir cambios
            subprocess.run("git add data/ screenshots/", shell=True, cwd=root)
            
            # 3. Solo commit si hay algo nuevo
            res = subprocess.run("git status --porcelain", shell=True, capture_output=True, text=True, cwd=root)
            if res.stdout.strip():
                subprocess.run('git commit -m "Sincronización automática de registros"', shell=True, cwd=root)
                # Intentar pull antes de push para evitar conflictos
                subprocess.run("git pull origin main --rebase", shell=True, cwd=root)
                subprocess.run("git push origin main", shell=True, cwd=root)
                print("[OmniFace] Sincronización con Git completada.")
            else:
                print("[OmniFace] Sin cambios locales para sincronizar.")
        except Exception as e:
            print(f"[OmniFace] Error en sincronización Git: {e}")

    threading.Thread(target=_sync, daemon=True).start()


def _add_col(conn, table: str, col: str, typedef: str):
    """Añade columna si no existe (SQLite no tiene IF NOT EXISTS en ALTER TABLE)."""
    existing = [r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]
    if col not in existing:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {typedef}")


def init_db():
    conn = get_connection()
    c = conn.cursor()

    # ── Cuentas de usuario (login web + desktop) ──────────────────────────────
    c.execute("""
        CREATE TABLE IF NOT EXISTS accounts (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            nombre        TEXT    NOT NULL,
            ap_paterno    TEXT    NOT NULL,
            ap_materno    TEXT    DEFAULT '',
            curp          TEXT    UNIQUE NOT NULL,
            fecha_nac     TEXT    NOT NULL,
            correo        TEXT    UNIQUE NOT NULL,
            password_hash TEXT    NOT NULL,
            role          TEXT    DEFAULT 'user',
            created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_renewal  TIMESTAMP
        )
    """)

    # ── Identidades faciales (sujetos reconocidos) ────────────────────────────
    c.execute("""
        CREATE TABLE IF NOT EXISTS identities (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            name         TEXT    NOT NULL,
            created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    # Migrar columnas de datos personales si no existen
    for col, typedef in [
        ("ap_paterno",   "TEXT DEFAULT ''"),
        ("ap_materno",   "TEXT DEFAULT ''"),
        ("curp",         "TEXT DEFAULT ''"),
        ("fecha_nac",    "TEXT DEFAULT ''"),
        ("correo",       "TEXT DEFAULT ''"),
        ("account_id",   "INTEGER"),
        ("last_renewal", "TIMESTAMP"),
    ]:
        _add_col(conn, "identities", col, typedef)

    # ── Muestras faciales (BLOBs JPEG 224×224 color) ─────────────────────────
    c.execute("""
        CREATE TABLE IF NOT EXISTS face_samples (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            identity_id INTEGER NOT NULL,
            face_data   BLOB    NOT NULL,
            FOREIGN KEY (identity_id) REFERENCES identities(id) ON DELETE CASCADE
        )
    """)

    # ── Historial de accesos / detecciones ────────────────────────────────────
    c.execute("""
        CREATE TABLE IF NOT EXISTS access_log (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            identity_id     INTEGER,
            identity_name   TEXT    NOT NULL,
            confidence      REAL,
            status          TEXT    NOT NULL,
            screenshot_path TEXT,
            camera_name     TEXT    DEFAULT '',
            timestamp       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    _add_col(conn, "access_log", "camera_name", "TEXT DEFAULT ''")

    # ── Cámaras configuradas ──────────────────────────────────────────────────
    c.execute("""
        CREATE TABLE IF NOT EXISTS cameras (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            name       TEXT    NOT NULL,
            source     TEXT    NOT NULL,
            active     INTEGER DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()

    # ── Migrar timestamps UTC → hora local (ejecución única) ─────────────────
    _migrate_utc_timestamps()

    # ── Cuenta admin por defecto (primera ejecución) ──────────────────────────
    _ensure_default_admin()


def _migrate_utc_timestamps():
    """
    Migración única: convierte timestamps UTC → hora local del dispositivo.
    Detecta si los registros están en UTC comparando el más reciente con
    la hora local actual. Si difiere más de 1 hora, aplica el offset.
    No vuelve a ejecutarse si los timestamps ya son locales.
    """
    conn = get_connection()
    try:
        last = conn.execute(
            "SELECT MAX(timestamp) FROM access_log"
        ).fetchone()[0]
        if not last:
            return  # Sin registros, nada que migrar

        last_dt    = datetime.strptime(last[:19], "%Y-%m-%d %H:%M:%S")
        local_now  = datetime.now()
        diff_secs  = (last_dt - local_now).total_seconds()

        # Si el timestamp más reciente es más de 1 hora en el "futuro" → está en UTC
        if diff_secs > 3600:
            utc_offset   = datetime.now().astimezone().utcoffset()
            offset_hours = int(utc_offset.total_seconds() / 3600)   # ej. -6
            sign         = "+" if offset_hours >= 0 else "-"
            abs_h        = abs(offset_hours)
            modifier     = f"{sign}{abs_h} hours"

            affected = conn.execute(
                f"UPDATE access_log SET timestamp = datetime(timestamp, ?)",
                (modifier,)
            ).rowcount
            conn.commit()
            print(f"[OmniFace] Timestamps migrados a hora local "
                  f"({modifier}): {affected} registros")
        # else: timestamps ya están en hora local → no hacer nada
    except Exception as e:
        print(f"[OmniFace] Error en migración de timestamps: {e}")
    finally:
        conn.close()


def fmt_local(ts_str: str) -> str:
    """
    Formatea un timestamp para mostrar en pantalla.
    Si por alguna razón llega un timestamp UTC (futuro > 1h), lo convierte
    automáticamente a hora local. Uso: en templates y en la GUI de escritorio.
    """
    if not ts_str:
        return "—"
    try:
        dt        = datetime.strptime(ts_str[:19], "%Y-%m-%d %H:%M:%S")
        local_now = datetime.now()
        # Guardia por si el valor es UTC
        if (dt - local_now).total_seconds() > 3600:
            utc_offset = datetime.now().astimezone().utcoffset()
            dt = dt + utc_offset
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return ts_str


def _ensure_default_admin():
    """Crea admin por defecto si no existe ninguna cuenta con role='admin'."""
    from .auth import create_account
    conn = get_connection()
    try:
        row = conn.execute("SELECT id FROM accounts WHERE role='admin' LIMIT 1").fetchone()
        if row:
            return  # Ya existe
    finally:
        conn.close()
    try:
        create_account(
            nombre="Admin", ap_paterno="OmniFace", ap_materno="",
            curp="ADMN000000HDFXXX00", fecha_nac="2000-01-01",
            correo="admin@omniface.local", password="admin1234",
            role="admin"
        )
        print("[OmniFace] Admin por defecto creado → correo: admin@omniface.local  pwd: admin1234")
    except ValueError:
        pass  # Ya existe (race condition o CURP duplicada)


# ── Identidades ───────────────────────────────────────────────────────────────

def save_identity(name: str, face_blobs: list) -> int:
    """Guarda identidad simple (compatibilidad hacia atrás)."""
    return save_identity_full(name, '', '', '', '', '', face_blobs)


def save_identity_full(nombre, ap_paterno, ap_materno, curp, fecha_nac,
                       correo, face_blobs, account_id=None) -> int:
    conn = get_connection()
    try:
        display_name = f"{nombre} {ap_paterno} {ap_materno}".strip()
        c = conn.cursor()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        c.execute("""
            INSERT INTO identities
                (name, ap_paterno, ap_materno, curp, fecha_nac, correo, account_id, last_renewal, created_at)
            VALUES (?,?,?,?,?,?,?,?,?)
        """, (display_name, ap_paterno, ap_materno, curp.upper(),
              fecha_nac, correo, account_id, now, now))
        identity_id = c.lastrowid
        c.executemany(
            "INSERT INTO face_samples (identity_id, face_data) VALUES (?,?)",
            [(identity_id, b) for b in face_blobs]
        )
        conn.commit()
        # Sincronizar con Git automáticamente
        git_sync()
        return identity_id
    finally:
        conn.close()


def get_all_identities() -> list[dict]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM identities ORDER BY created_at DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_identity_full(identity_id: int) -> dict | None:
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM identities WHERE id = ?", (identity_id,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def get_identity_by_account(account_id: int) -> dict | None:
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM identities WHERE account_id = ? ORDER BY created_at DESC LIMIT 1",
        (account_id,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def delete_identity(identity_id: int):
    conn = get_connection()
    conn.execute("DELETE FROM identities WHERE id = ?", (identity_id,))
    conn.commit()
    conn.close()


def get_all_face_samples() -> list[tuple]:
    conn = get_connection()
    rows = conn.execute("""
        SELECT fs.identity_id, i.name, fs.face_data
        FROM face_samples fs
        JOIN identities i ON fs.identity_id = i.id
    """).fetchall()
    conn.close()
    return [(r["identity_id"], r["name"], bytes(r["face_data"])) for r in rows]


def update_identity_renewal(identity_id: int):
    conn = get_connection()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn.execute(
        "UPDATE identities SET last_renewal = ? WHERE id = ?",
        (now, identity_id)
    )
    conn.commit()
    conn.close()


# ── Logs ──────────────────────────────────────────────────────────────────────

def log_access(identity_name, status, confidence=None,
               identity_id=None, screenshot_path=None, camera_name=""):
    conn = get_connection()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn.execute("""
        INSERT INTO access_log
            (identity_id, identity_name, confidence, status, screenshot_path, camera_name, timestamp)
        VALUES (?,?,?,?,?,?,?)
    """, (identity_id, identity_name, confidence, status, screenshot_path, camera_name, now))
    conn.commit()
    conn.close()
    # Notificar al pusher periódico para que incluya esta detección en el próximo push
    try:
        from .sync import mark_detection_pending
        mark_detection_pending()
    except Exception:
        pass  # Si sync no está disponible (tests/scripts) no es crítico


def get_access_logs(limit=100, identity_id=None) -> list[dict]:
    conn = get_connection()
    if identity_id:
        rows = conn.execute("""
            SELECT * FROM access_log WHERE identity_id = ?
            ORDER BY timestamp DESC LIMIT ?
        """, (identity_id, limit)).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM access_log ORDER BY timestamp DESC LIMIT ?", (limit,)
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_stats() -> dict:
    conn = get_connection()
    total_ids   = conn.execute("SELECT COUNT(*) FROM identities").fetchone()[0]
    total_logs  = conn.execute("SELECT COUNT(*) FROM access_log").fetchone()[0]
    known_today = conn.execute(
        "SELECT COUNT(*) FROM access_log WHERE status='known' AND DATE(timestamp)=DATE('now')"
    ).fetchone()[0]
    unk_today   = conn.execute(
        "SELECT COUNT(*) FROM access_log WHERE status='unknown' AND DATE(timestamp)=DATE('now')"
    ).fetchone()[0]
    conn.close()
    return {
        "total_identities": total_ids,
        "total_accesses":   total_logs,
        "known_today":      known_today,
        "unknown_today":    unk_today,
    }


def get_user_stats(identity_id: int) -> dict:
    """Estadísticas personales del usuario (sus propias detecciones)."""
    conn = get_connection()
    total = conn.execute(
        "SELECT COUNT(*) FROM access_log WHERE identity_id = ?", (identity_id,)
    ).fetchone()[0]
    lugares = conn.execute(
        "SELECT COUNT(DISTINCT camera_name) FROM access_log WHERE identity_id = ?",
        (identity_id,)
    ).fetchone()[0]
    hoy = conn.execute(
        "SELECT COUNT(*) FROM access_log WHERE identity_id = ? AND DATE(timestamp)=DATE('now')",
        (identity_id,)
    ).fetchone()[0]
    ultima = conn.execute(
        "SELECT timestamp FROM access_log WHERE identity_id = ? ORDER BY timestamp DESC LIMIT 1",
        (identity_id,)
    ).fetchone()
    conn.close()
    return {
        "total":       total,
        "lugares":     lugares,
        "hoy":         hoy,
        "ultima":      ultima[0] if ultima else None,
    }


def get_deduped_logs(identity_id: int, limit: int = 10) -> list[dict]:
    """
    Últimas detecciones deduplicadas por cámara:
    una sola fila por cámara (la más reciente).
    Ordenadas por ese timestamp más reciente DESC.
    """
    conn = get_connection()
    rows = conn.execute("""
        SELECT a.*
        FROM access_log a
        INNER JOIN (
            SELECT camera_name, MAX(timestamp) AS max_ts
            FROM access_log
            WHERE identity_id = ?
            GROUP BY camera_name
        ) g ON a.camera_name = g.camera_name AND a.timestamp = g.max_ts
        WHERE a.identity_id = ?
        ORDER BY a.timestamp DESC
        LIMIT ?
    """, (identity_id, identity_id, limit)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ── Cámaras ───────────────────────────────────────────────────────────────────

def get_cameras() -> list[dict]:
    conn = get_connection()
    rows = conn.execute("SELECT * FROM cameras ORDER BY id").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def add_camera(name: str, source: str) -> int:
    conn = get_connection()
    c = conn.cursor()
    c.execute("INSERT INTO cameras (name, source, active) VALUES (?,?,1)", (name, str(source)))
    conn.commit()
    cam_id = c.lastrowid
    conn.close()
    return cam_id


def remove_camera(cam_id: int):
    conn = get_connection()
    conn.execute("DELETE FROM cameras WHERE id = ?", (cam_id,))
    conn.commit()
    conn.close()


def toggle_camera(cam_id: int, active: bool):
    conn = get_connection()
    conn.execute("UPDATE cameras SET active = ? WHERE id = ?", (int(active), cam_id))
    conn.commit()
    conn.close()
