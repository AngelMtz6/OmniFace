import sqlite3
import json
import os

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'omniface.db')


def get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS identities (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT    NOT NULL,
            photo_path  TEXT,
            encodings   TEXT    NOT NULL,
            created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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


def save_identity(name, encodings, photo_path=None):
    conn = get_connection()
    c = conn.cursor()
    c.execute(
        'INSERT INTO identities (name, encodings, photo_path) VALUES (?, ?, ?)',
        (name, json.dumps([e.tolist() for e in encodings]), photo_path)
    )
    identity_id = c.lastrowid
    conn.commit()
    conn.close()
    return identity_id


def get_all_identities():
    conn = get_connection()
    rows = conn.execute(
        'SELECT id, name, photo_path, created_at FROM identities ORDER BY created_at DESC'
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_all_encodings():
    conn = get_connection()
    rows = conn.execute('SELECT id, name, encodings FROM identities').fetchall()
    conn.close()
    result = []
    for r in rows:
        for enc in json.loads(r['encodings']):
            result.append({'id': r['id'], 'name': r['name'], 'encoding': enc})
    return result


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
