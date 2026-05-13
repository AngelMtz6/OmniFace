# OmniFace-1 — Stack Completo

> Documento de referencia para el equipo. Todo lo que se usa en el proyecto.

---

## Lenguaje

| Elemento | Detalle |
|----------|---------|
| **Python** | 3.14 |
| **Entorno virtual** | `.venv\` — activar con `.venv\Scripts\activate` |
| **Entry point escritorio** | `run.bat` ó `.venv\Scripts\python.exe main_gui.py` |
| **Entry point web (HTTP)** | `.venv\Scripts\python.exe run.py` → `localhost:5000` |
| **Entry point web (HTTPS)** | `.venv\Scripts\python.exe run_ngrok.py` → URL pública ngrok |
| **Auto-relanzador** | `main_gui.py` detecta si corre sin venv y se relanza solo |

---

## Librerías principales

| Librería | Versión | Para qué se usa |
|----------|---------|-----------------|
| `insightface` | 0.7.3 | Framework de reconocimiento facial Deep Learning (RetinaFace + ArcFace) |
| `onnxruntime` | 1.26.0 | Runtime de modelos ONNX — GPU vía `CUDAExecutionProvider` si CUDA 12 disponible |
| `opencv-contrib-python` | 4.10.0 | Captura de cámara, CLAHE, transformaciones de imagen, dibujo de bounding boxes |
| `numpy` | latest | Arrays de imágenes, operaciones de embeddings (coseno, normalización) |
| `customtkinter` | 5.2.0 | GUI de escritorio — ventanas, botones, sidebar, labels de video |
| `Pillow` (PIL) | 10.0.0 | Convertir frames OpenCV → formato compatible con tkinter (`ImageTk.PhotoImage`) |
| `pystray` | 0.19.0 | Ícono en bandeja del sistema (minimizar a tray) |
| `Flask` | 3.x | Servidor web — rutas, sesiones, Jinja2 |
| `Werkzeug` | — | Hashing de contraseñas (`generate_password_hash` / `check_password_hash`) |
| `pyngrok` | latest | Túnel HTTPS automático — permite cámara web en dispositivos remotos |

---

## Módulos de Python (stdlib)

| Módulo | Uso |
|--------|-----|
| `sqlite3` | Base de datos local (identidades, muestras faciales, historial de acceso) |
| `threading` | Procesamiento en fondo: inferencia, retrain, gestión de locks |
| `concurrent.futures` | — (removido, reemplazado por threading directo) |
| `ctypes` | Verificar que `cublasLt64_12.dll` carga antes de activar CUDA |
| `time` | Cooldown de logs, pausas entre pasos de registro |
| `os` / `subprocess` | Rutas, directorios, consulta de GPU con `nvidia-smi` |
| `tkinter` | Base de CustomTkinter + `Label` para video en vivo |

---

## Deep Learning — InsightFace

| Componente | Descripción |
|-----------|-------------|
| **FaceAnalysis** | Clase principal — orquesta detección + reconocimiento |
| **RetinaFace** | Detector de caras — maneja frente, perfil y ángulos intermedios nativamente. Reemplaza los 3 Haar cascades anteriores |
| **ArcFace (w600k_r50 / w600k_mbf)** | Reconocedor facial — produce embeddings de 512 dimensiones. Similitud coseno en lugar de distancia LBPH |
| **buffalo_l** | Modelo completo (ResNet50) — usado cuando GPU CUDA 12 disponible. `det_size=(640,480)` |
| **buffalo_sc** | Modelo compacto (MobileFaceNet) — usado en CPU. `det_size=(256,192)` (ambas dims múltiplo de 32) |
| **Modelos** | Descargados automáticamente en `~/.insightface/models/` en primer uso |

### Requisito GPU
- `onnxruntime` 1.26.0 con `CUDAExecutionProvider`
- **CUDA Toolkit 12.x** — proporciona `cublasLt64_12.dll`
- Sin CUDA Toolkit: cae a CPU automáticamente con buffalo_sc

---

## OpenCV — Componentes usados

| Componente | Descripción |
|-----------|-------------|
| `cv2.VideoCapture(0, cv2.CAP_DSHOW)` | Captura de cámara en Windows |
| `cv2.createCLAHE` | Normalización de iluminación en data augmentation |
| `cv2.flip` | Espejo horizontal para augmentation |
| `cv2.convertScaleAbs` | Variantes bright/dark para augmentation |
| `cv2.imencode / imdecode` | Serializar/deserializar crops JPEG (almacenamiento en SQLite) |
| `cv2.resize` | Escalar crops a 224×224 (registro) y display |
| `cv2.rectangle / putText` | Dibujar bounding boxes y etiquetas en el video |
| `cv2.copyMakeBorder` | Padding de crops 112×112 → 224×224 para retrain |

---

## Arquitectura de archivos

```
OmniFace-1/
├── main_gui.py          ← App escritorio (CustomTkinter) — monitor, registro, gestión
├── run.py               ← Entry point Flask (HTTP localhost:5000)
├── run_ngrok.py         ← Entry point Flask + túnel HTTPS ngrok
├── run.bat              ← Lanzador escritorio con venv correcto
├── run_ngrok.bat        ← Lanzador web con ngrok (doble clic)
│
├── app/
│   ├── __init__.py      ← App factory Flask (create_app)
│   ├── routes.py        ← Rutas web: login, registro, dashboard, perfil, historial, API
│   ├── auth.py          ← Autenticación web — create_account, login, reset password
│   ├── recognition.py   ← Motor InsightFace (RetinaFace + ArcFace + retrain)
│   ├── camera.py        ← Singleton VideoCamera — get_frame() thread-safe
│   ├── database.py      ← SQLite CRUD — identidades, muestras, logs, stats de usuario
│   ├── alerts.py        ← AlertManager — capturas de desconocidos
│   └── sync.py          ← Sincronización Git:
│                             push_db()              — commit + push en hilo de fondo
│                             pull_db_identities()   — importa solo identidades nuevas
│                             start_detection_pusher() — push automático cada 10 s
│                             mark_detection_pending() — llamado por log_access()
│
├── templates/           ← Jinja2 — Bootstrap 5 dark theme
│   ├── base.html
│   ├── login.html
│   ├── register.html
│   ├── dashboard.html        ← Stats personales + detecciones deduplicadas por cámara
│   ├── profile.html
│   ├── my_history.html       ← Historial completo con barra de confianza
│   ├── register_face.html    ← Captura 60 frames via webcam → POST /api/register_face
│   └── forgot_password.html
│
├── static/
│   └── screenshots/     ← Capturas automáticas de desconocidos
│
├── data/
│   └── omniface.db      ← Base de datos SQLite (generada automáticamente)
│
├── assets/
│   └── logo.png
├── CLAUDE.md            ← Knowledge Graph del proyecto (para Claude)
├── STACK.md             ← Este archivo
└── requirements.txt
```

---

## Base de datos (SQLite)

```sql
-- Cuentas de usuario (login web + desktop)
accounts (
    id            INTEGER PRIMARY KEY,
    nombre        TEXT NOT NULL,
    ap_paterno    TEXT NOT NULL,
    ap_materno    TEXT DEFAULT '',
    curp          TEXT UNIQUE NOT NULL,
    fecha_nac     TEXT NOT NULL,
    correo        TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    role          TEXT DEFAULT 'user',   -- 'user' | 'admin'
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)

-- Personas registradas con datos personales
identities (
    id           INTEGER PRIMARY KEY,
    name         TEXT NOT NULL,          -- nombre completo display
    ap_paterno   TEXT DEFAULT '',
    ap_materno   TEXT DEFAULT '',
    curp         TEXT DEFAULT '',
    fecha_nac    TEXT DEFAULT '',
    correo       TEXT DEFAULT '',
    account_id   INTEGER,               -- FK → accounts.id (NULL si registro desktop)
    last_renewal TIMESTAMP,
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)

-- Muestras faciales por persona (BLOBs JPEG 224×224 color)
face_samples (
    id          INTEGER PRIMARY KEY,
    identity_id INTEGER NOT NULL REFERENCES identities(id) ON DELETE CASCADE,
    face_data   BLOB NOT NULL
)

-- Historial de detecciones
access_log (
    id              INTEGER PRIMARY KEY,
    identity_id     INTEGER,
    identity_name   TEXT NOT NULL,
    confidence      REAL,               -- porcentaje 0-100 (NO multiplicar × 100 en templates)
    status          TEXT NOT NULL,      -- 'known' | 'unknown'
    screenshot_path TEXT,
    camera_name     TEXT DEFAULT '',
    timestamp       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)

-- Cámaras configuradas en la app de escritorio
cameras (
    id         INTEGER PRIMARY KEY,
    name       TEXT NOT NULL,
    source     TEXT NOT NULL,
    active     INTEGER DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
```

---

## Pipeline de reconocimiento (tiempo real)

```
Frame de cámara (640×480)
    │
    ▼  [hilo de fondo — no bloquea GUI]
InsightFace FaceAnalysis.get(frame)
    ├── RetinaFace detector
    │   └── Redimensiona a det_size → genera anchors → NMS interno
    │       Detecta frente, perfil y ángulos intermedios en una sola pasada
    └── Por cada cara detectada:
        ├── norm_crop() → cara alineada 112×112 (usando landmarks faciales)
        └── ArcFace.get_feat() → embedding 512-dim
    │
    ▼
_identify(embedding)
    └── Similitud coseno vs embeddings promedio de cada identidad
    └── sim ≥ 0.40  → conocido,  confidence = sim × 100
    └── sim < 0.40  → Desconocido
```

---

## Pipeline de entrenamiento (retrain)

```
DB face_samples (BLOBs JPEG 224×224)
    │
    ▼ (por cada muestra)
cv2.copyMakeBorder si 112×112 → pad a 224×224
    │
    ▼
_augment()  →  [ original, flip, bright×1.25, dark×0.75 ]  (GPU: 4×, CPU: 1×)
    │
    ▼ (por cada variante)
InsightFace FaceAnalysis.get(variante)
    └── RetinaFace detecta la cara en el crop con padding
    └── ArcFace → embedding 512-dim
    │
    ▼ (por identidad)
np.mean(embeddings, axis=0) → embedding promedio normalizado
    │
    ▼
_known[identity_id] = {'name': ..., 'embedding': mean_emb}
```

---

## GUI escritorio — Vistas implementadas

| Vista | Estado | Descripción |
|-------|--------|-------------|
| `monitoring` | ✅ | Video en vivo con bounding boxes y nombre/confianza. Inferencia en hilo de fondo |
| `registration` | ✅ | Registro multi-ángulo automático (6 pasos × 10 muestras = 60 crops 224×224) |
| `database` | ✅ | Lista de identidades con opción de eliminar |
| Historial | ✅ | Tabla de logs con confianza, cámara y estado |

---

## Portal web — Rutas implementadas

| Ruta | Método | Descripción |
|------|--------|-------------|
| `/` | GET | Redirect → dashboard si autenticado, login si no |
| `/login` | GET/POST | Autenticación con correo + contraseña |
| `/register` | GET/POST | Crear cuenta (CURP, datos personales) |
| `/forgot-password` | GET/POST | Solicitud de reset de contraseña |
| `/logout` | GET | Cerrar sesión |
| `/dashboard` | GET | Stats personales + últimas detecciones deduplicadas |
| `/profile` | GET | Perfil completo + estado del registro facial |
| `/my-history` | GET | Historial completo de detecciones |
| `/register_face` | GET | Página de captura facial (60 frames webcam) |
| `/api/register_face` | POST | Recibe frames base64 → guarda identidad → push Git |

---

## Sincronización Git

| Evento | Función | Detalle |
|--------|---------|---------|
| Registro facial (web) | `push_db()` | Inmediato, hilo de fondo |
| Detecciones (desktop) | `start_detection_pusher()` | Batch cada 10 s — `mark_detection_pending()` en `log_access()` |
| Eliminar identidad (desktop) | `push_db()` | Inmediato |
| Botón Sincronizar Nube | `pull_db_identities()` | Solo importa identidades nuevas — NO reemplaza DB completo |

> **Conflictos de binary merge:** Desktop nunca hace `git pull` del DB completo.
> Lee el remoto con `git show origin/main:data/omniface.db` en archivo temporal
> y hace `INSERT OR IGNORE` a nivel de filas SQLite.

---

## Parámetros clave ajustables

| Parámetro | Valor actual | Archivo |
|-----------|-------------|---------|
| `COSINE_THRESHOLD` | `0.40` | `recognition.py` |
| `PROCESS_EVERY_N` | `1` (GPU) / `3` (CPU) | `recognition.py` |
| `LOG_COOLDOWN` | `5` segundos | `recognition.py` |
| `_MODEL_NAME` | `buffalo_l` (GPU) / `buffalo_sc` (CPU) | `recognition.py` |
| `_DET_SIZE` | `(640,480)` (GPU) / `(256,192)` (CPU) | `recognition.py` |
| Muestras por paso | `10` (× 6 pasos = 60 total) | `main_gui.py` |
| Resolución cámara | `640 × 480` | `camera.py` |
| Crop de registro | `224 × 224` color JPEG | `main_gui.py` |

---

## Diferencias clave vs versión anterior (LBPH)

| Aspecto | Antes (LBPH) | Ahora (ArcFace) |
|---------|-------------|----------------|
| Detección | 3 Haar cascades en paralelo (frente + 2 perfiles) | RetinaFace — un solo modelo, todos los ángulos |
| Reconocimiento | LBPH + distancia euclidiana | ArcFace + similitud coseno |
| Precisión esperada | ~50-80% | ~95-99% |
| Robustez a iluminación | Baja — requería CLAHE | Alta — ArcFace entrenado con variaciones |
| Robustez a ángulos | Media — requería cascades múltiples | Alta — RetinaFace maneja ±90° |
| Muestras necesarias | 180 (30×6) | 60 (10×6) |
| GPU | No (OpenCV sin CUDA en PyPI) | Sí (onnxruntime CUDAExecutionProvider) |

---

## Herramientas de desarrollo

| Herramienta | Uso |
|-------------|-----|
| **Git** | Control de versiones + transporte de DB entre dispositivos |
| **GitHub** — `AngelMtz6/OmniFace` | Repositorio remoto, rama `main` |
| **VS Code** | Editor — `.vscode/launch.json` configurado para venv |
| **Claude** | Asistente de desarrollo (Vibe Coding) |
| **nvidia-smi** | Verificar GPU disponible |
| **CUDA Toolkit 12.x** | Requerido para activar GPU (proporciona `cublasLt64_12.dll`) |
| **ngrok** | Túnel HTTPS para probar cámara web fuera de localhost |
