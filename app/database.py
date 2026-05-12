import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'omniface.db')


def get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=20)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    # Verificar si la base de datos tiene el esquema antiguo
    if os.path.exists(DB_PATH):
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(identities)")
            columns = [column[1] for column in cursor.fetchall()]
            conn.close()
            
            if 'encodings' in columns:
                print("DEBUG: Detectado esquema antiguo. Recreando base de datos...")
                os.remove(DB_PATH)
        except Exception as e:
            print(f"DEBUG: Error verificando esquema: {e}")

    conn = get_connection()
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS identities (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            name       TEXT    NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    # Almacena cada muestra facial como JPEG 100x100 en escala de grises
    c.execute('''
        CREATE TABLE IF NOT EXISTS face_samples (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            identity_id INTEGER NOT NULL,
            face_data   BLOB    NOT NULL,
            FOREIGN KEY (identity_id) REFERENCES identities(id) ON DELETE CASCADE
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS access_log (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            identity_id     INTEGER,
            identity_name   TEXT    NOT NULL,
            confidence      REAL,
            status          TEXT    NOT NULL,
            screenshot_path TEXT,
            timestamp       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()


def save_identity(name, face_blobs):
    """Crea identidad y guarda las muestras faciales. Retorna el nuevo id."""
    conn = get_connection()
    c = conn.cursor()
    c.execute('INSERT INTO identities (name) VALUES (?)', (name,))
    identity_id = c.lastrowid
    c.executemany(
        'INSERT INTO face_samples (identity_id, face_data) VALUES (?, ?)',
        [(identity_id, blob) for blob in face_blobs]
    )
    conn.commit()
    conn.close()
    return identity_id


def get_all_identities():
    conn = get_connection()
    rows = conn.execute(
        'SELECT id, name, created_at FROM identities ORDER BY created_at DESC'
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_all_face_samples():
    """Retorna lista de (identity_id, name, face_data_bytes)."""
    conn = get_connection()
    rows = conn.execute('''
        SELECT fs.identity_id, i.name, fs.face_data
        FROM face_samples fs
        JOIN identities i ON fs.identity_id = i.id
    ''').fetchall()
    conn.close()
    return [(r['identity_id'], r['name'], bytes(r['face_data'])) for r in rows]


def delete_identity(identity_id):
    conn = get_connection()
    conn.execute('DELETE FROM identities WHERE id = ?', (identity_id,))
    conn.commit()
    conn.close()


def log_access(identity_name, status, confidence=None, identity_id=None, screenshot_path=None):
    conn = get_connection()
    conn.execute(
        'INSERT INTO access_log (identity_id, identity_name, confidence, status, screenshot_path) '
        'VALUES (?, ?, ?, ?, ?)',
        (identity_id, identity_name, confidence, status, screenshot_path)
    )
    conn.commit()
    conn.close()


def get_access_logs(limit=100):
    conn = get_connection()
    rows = conn.execute(
        'SELECT * FROM access_log ORDER BY timestamp DESC LIMIT ?', (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_stats():
    conn = get_connection()
    total_identities = conn.execute('SELECT COUNT(*) FROM identities').fetchone()[0]
    total_accesses   = conn.execute('SELECT COUNT(*) FROM access_log').fetchone()[0]
    known_today      = conn.execute(
        "SELECT COUNT(*) FROM access_log WHERE status='known'   AND DATE(timestamp)=DATE('now')"
    ).fetchone()[0]
    unknown_today    = conn.execute(
        "SELECT COUNT(*) FROM access_log WHERE status='unknown' AND DATE(timestamp)=DATE('now')"
    ).fetchone()[0]
    conn.close()
    return {
        'total_identities': total_identities,
        'total_accesses':   total_accesses,
        'known_today':      known_today,
        'unknown_today':    unknown_today,
    }
