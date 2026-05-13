import hashlib
import os
import sqlite3
from .database import get_connection


def _hash(password: str) -> str:
    salt = os.urandom(16).hex()
    h    = hashlib.sha256(f"{salt}:{password}".encode()).hexdigest()
    return f"{salt}:{h}"


def _verify(password: str, stored: str) -> bool:
    try:
        salt, h = stored.split(":", 1)
        return hashlib.sha256(f"{salt}:{password}".encode()).hexdigest() == h
    except Exception:
        return False


def create_account(nombre, ap_paterno, ap_materno, curp, fecha_nac,
                   correo, password, role="user") -> int:
    """Crea cuenta. Lanza ValueError si CURP o correo ya existen."""
    conn = get_connection()
    try:
        c = conn.cursor()
        c.execute(
            """INSERT INTO accounts
               (nombre, ap_paterno, ap_materno, curp, fecha_nac, correo, password_hash, role)
               VALUES (?,?,?,?,?,?,?,?)""",
            (nombre, ap_paterno, ap_materno, curp.upper(),
             fecha_nac, correo.lower(), _hash(password), role)
        )
        conn.commit()
        return c.lastrowid
    except sqlite3.IntegrityError as e:
        msg = str(e).lower()
        if "curp"   in msg: raise ValueError("CURP ya registrada")
        if "correo" in msg: raise ValueError("Correo ya registrado")
        raise ValueError(str(e))
    finally:
        conn.close()


def login(correo: str, password: str) -> dict | None:
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM accounts WHERE correo = ?", (correo.lower(),)
        ).fetchone()
        if row and _verify(password, row["password_hash"]):
            return dict(row)
        return None
    finally:
        conn.close()


def get_account(account_id: int) -> dict | None:
    conn = get_connection()
    try:
        row = conn.execute("SELECT * FROM accounts WHERE id = ?", (account_id,)).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def request_password_reset(correo: str) -> bool:
    """Simulado — en producción enviaría correo con token."""
    conn = get_connection()
    try:
        row = conn.execute("SELECT id FROM accounts WHERE correo = ?",
                           (correo.lower(),)).fetchone()
        if row:
            print(f"[OmniFace Auth] Recuperación simulada para {correo}")
            return True
        return False
    finally:
        conn.close()
