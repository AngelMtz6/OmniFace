# OmniFace — Knowledge Graph del Proyecto

> Este archivo es el mapa de contexto del proyecto. Léelo primero antes de tocar cualquier archivo.

---

## Identidad del proyecto
- **Nombre:** OmniFace
- **Evento:** Hackatec 2026
- **Tipo:** Sistema de reconocimiento facial en tiempo real (escritorio + web)
- **Repo:** https://github.com/AngelMtz6/OmniFace
- **Directorio activo:** `C:\Users\angel\Desktop\Hackatec\OmniFace`

---

## Stack actual (después de pivote)

| Capa | Tecnología | Por qué |
|------|-----------|---------|
| Reconocimiento | OpenCV LBPH (`cv2.face.LBPHFaceRecognizer`) | Reemplazó `face_recognition`/dlib, sin dependencias externas |
| Detección | Haar Cascade (`haarcascade_frontalface_default.xml`) | Incluido en OpenCV, rápido en CPU |
| Captura | `VideoCamera` auto-detecta: default → `CAP_DSHOW` → índice 1 | Robusto ante error MSMF `-1072875772` |
| Storage | SQLite — blobs JPEG 100×100 grises | Sin archivos externos, portátil |
| Backend web | Flask 3.x + SSE, `use_reloader=False` | Evita doble instancia de cámara en modo debug |
| Frontend | HTML/CSS/JS vanilla | Sin frameworks, fácil de editar |

---

## Árbol de archivos y responsabilidades

```
OmniFace/
│
├── run.py                      ← ENTRADA. `python run.py` arranca en :5000
│
├── app/
│   ├── __init__.py             ← Flask app factory + init_db()
│   ├── camera.py               ← Singleton VideoCamera (CAP_DSHOW, thread-safe)
│   ├── database.py             ← ÚNICA fuente de verdad para SQLite
│   ├── recognition.py          ← Motor LBPH: detección + identificación + dibujo
│   ├── alerts.py               ← Contador de desconocidos → alerta + screenshot
│   └── routes.py               ← Todos los endpoints Flask + generador MJPEG
│
├── templates/
│   ├── base.html               ← Navbar + SSE global de alertas
│   ├── index.html              ← /        Monitor en vivo + stats + últimos eventos
│   ├── register.html           ← /register  Enrollamiento con captura múltiple
│   └── admin.html              ← /admin    Panel completo: identidades + historial
│
├── static/
│   ├── css/style.css           ← Dark theme completo, sin dependencias externas
│   └── js/app.js               ← Solo SSE global de navbar
│
├── data/omniface.db            ← Auto-generado. NO commitear (en .gitignore)
├── screenshots/                ← Auto-generado. NO commitear (en .gitignore)
├── test_cam.py                 ← Diagnóstico de backends de cámara
└── CLAUDE.md                   ← Este archivo ← LEE PRIMERO
```

---

## Schema de base de datos

```sql
identities (
  id         INTEGER PK AUTOINCREMENT,
  name       TEXT NOT NULL,
  created_at TIMESTAMP DEFAULT NOW
)

face_samples (                        -- Una fila por imagen capturada
  id          INTEGER PK AUTOINCREMENT,
  identity_id INTEGER FK → identities.id ON DELETE CASCADE,
  face_data   BLOB                    -- JPEG 100×100 escala de grises
)

access_log (
  id              INTEGER PK,
  identity_id     INTEGER FK (nullable = desconocido),
  identity_name   TEXT,
  confidence      REAL,               -- 0-100, mayor = más seguro
  status          TEXT,               -- 'known' | 'unknown'
  screenshot_path TEXT,               -- solo para desconocidos
  timestamp       TIMESTAMP
)
```

---

## API endpoints

| Método | Ruta | Función en routes.py | Descripción |
|--------|------|---------------------|-------------|
| GET | `/` | `index()` | Página monitor en vivo |
| GET | `/register` | `register()` | Página de enrollamiento |
| GET | `/admin` | `admin()` | Panel de administración |
| GET | `/video_feed` | `_generate_frames()` | Stream MJPEG procesado |
| GET | `/capture_frame` | `capture_frame()` | Frame actual como base64 JSON |
| POST | `/enroll` | `enroll()` | Registrar nueva identidad |
| GET | `/identities` | `identities()` | Lista de identidades |
| DELETE | `/identity/<id>` | `delete()` | Eliminar identidad |
| GET | `/logs` | `logs()` | Historial de accesos |
| GET | `/stats` | `stats()` | Contadores globales |
| GET | `/alert_status` | `alert_status()` | Estado de alerta actual |
| GET | `/events` | `events()` | SSE → `{"alert": bool}` |

---

## Flujo de datos principal

```
VideoCamera (hilo daemon)
  ├── Auto-detecta backend: default → CAP_DSHOW → índice 1
  ├── Aplica flip horizontal (efecto espejo)
  └── get_frame() → (frame: ndarray, frame_id: int)
        ↓
routes._generate_frames()
  ├── Descarta frames duplicados via frame_id
  └── Llama engine.process_frame(frame)
        ↓
RecognitionEngine.process_frame()
  ├── Haar Cascade → rects de caras
  ├── LBPH.predict() → (label, raw_confidence)
  ├── _draw() → bounding boxes + texto en BGR
  └── _try_log() → access_log + screenshot si desconocido
        ↓
MJPEG stream → <img src="/video_feed">

AlertManager.update() → SSE /events → {"alert": bool} → overlay rojo en frontend
```

---

## Parámetros clave para ajustar

| Archivo | Variable | Valor actual | Efecto |
|---------|----------|-------------|--------|
| recognition.py | `LBPH_THRESH` | `85` | ↓ = más estricto, ↑ = más permisivo |
| recognition.py | `PROCESS_EVERY_N` | `3` | ↑ = más rápido, menos preciso |
| recognition.py | `LOG_COOLDOWN` | `5` seg | Tiempo mínimo entre logs de la misma persona |
| recognition.py | `FACE_SIZE` | `(100,100)` | Resolución de normalización de rostros |
| alerts.py | `UNKNOWN_THRESHOLD` | `3` | Detecciones seguidas para disparar alerta |
| alerts.py | `ALERT_RESET_SECONDS` | `10` | Tiempo de enfriamiento de alertas |

---

## Historial de decisiones técnicas

1. **`face_recognition` → OpenCV LBPH** — `dlib` no instalaba en Python 3.14 en Windows. LBPH es parte de `opencv-contrib-python`, cero dependencias extras.
2. **`CAP_DSHOW` en lugar de `CAP_MSMF`** — Error MSMF `-1072875772` en laptops Windows. DirectShow es más compatible.
3. **Blobs JPEG en SQLite** — Simplifica portabilidad: un solo archivo `.db` contiene todo. No hay carpeta de imágenes que gestionar.
4. **SSE en lugar de WebSocket** — Más simple para push server→cliente. Solo se necesita notificar `{alert: bool}`.

---

## Estado actual al último commit

- [x] Estructura completa del proyecto
- [x] Motor LBPH funcional (detección + reconocimiento + dibujo)
- [x] Sistema de enrollamiento vía web
- [x] Streaming MJPEG en tiempo real con deduplicación por `frame_id`
- [x] Alertas SSE para desconocidos
- [x] Panel de administración
- [x] `VideoCamera` auto-detecta backend (default / DSHOW / índice 1)
- [x] Flip horizontal (espejo) en captura
- [x] `use_reloader=False` en Flask para evitar doble instancia de cámara
- [ ] Detección de cara parcial (pendiente)
- [ ] Modo demo con datos pre-cargados
- [ ] README desactualizado (aún menciona dlib — no se usa)

---

## Cómo correr el proyecto

```powershell
cd "C:\Users\angel\Desktop\Hackatec\OmniFace"
pip install opencv-contrib-python flask numpy Pillow
python run.py
# → http://localhost:5000
```
