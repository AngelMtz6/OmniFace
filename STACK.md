# OmniFace-1 — Stack Completo

> Documento de referencia para el equipo. Todo lo que se usa en el proyecto.

---

## Lenguaje

| Elemento | Detalle |
|----------|---------|
| **Python** | 3.10+ |
| **Entorno virtual** | `.venv\` — activar con `.venv\Scripts\activate` |
| **Entry point** | `python main_gui.py` |

---

## Librerías (requirements.txt)

| Librería | Versión mínima | Para qué se usa |
|----------|---------------|-----------------|
| `opencv-contrib-python` | 4.9.0 | Captura de cámara, detección de caras (Haar), reconocimiento LBPH, CLAHE, transformaciones de imagen |
| `numpy` | 1.24.0 | Arrays de imágenes, labels para entrenamiento LBPH |
| `customtkinter` | 5.2.0 | GUI de escritorio — ventanas, botones, sidebar, labels de video |
| `Pillow` (PIL) | 10.0.0 | Convertir frames OpenCV → formato compatible con tkinter (`ImageTk.PhotoImage`) |
| `pystray` | 0.19.0 | Ícono en bandeja del sistema (minimizar a tray) |
| `flask` | 3.0.0 | **Residuo web — NO se usa en desktop** (`routes.py` no activo) |

---

## Módulos de Python (stdlib)

| Módulo | Uso |
|--------|-----|
| `sqlite3` | Base de datos local (identidades, muestras faciales, historial de acceso) |
| `threading` | Lock para acceso seguro al reconocedor desde múltiples hilos |
| `time` | Cooldown de logs, pausas entre pasos de registro |
| `os` | Rutas de archivos, crear directorios (`data/`, `screenshots/`) |
| `tkinter` | Base de CustomTkinter + `Label` para video en vivo |

---

## OpenCV — Componentes usados

| Componente | Descripción |
|-----------|-------------|
| `cv2.VideoCapture(0, cv2.CAP_DSHOW)` | Captura de cámara en Windows (CAP_DSHOW evita error MSMF) |
| `cv2.CascadeClassifier` | Detector de caras con modelos Haar |
| `haarcascade_frontalface_default.xml` | Detección de cara frontal (0°) |
| `haarcascade_profileface.xml` | Detección de cara de perfil (90°) — se corre x2: normal + frame volteado |
| `haarcascade_eye_tree_eyeglasses.xml` | Detección de ojos para alineación de cara |
| `cv2.face.LBPHFaceRecognizer_create` | Reconocedor facial LBPH (`radius=2, neighbors=8, grid_x=8, grid_y=8`) |
| `cv2.createCLAHE` | Normalización de iluminación (`clipLimit=2.0, tileGridSize=(8,8)`) |
| `cv2.warpAffine` | Rotación de imagen para alinear ojos horizontalmente |
| `cv2.flip` | Espejo horizontal — para perfil izquierdo y augmentation |
| `cv2.convertScaleAbs` | Variantes bright/dark para augmentation |
| `cv2.imencode / imdecode` | Serializar/deserializar imágenes a JPEG (almacenamiento en SQLite) |
| `cv2.resize` | Escalar caras a 150×150 antes de entrenar/predecir |
| `cv2.rectangle / putText` | Dibujar bounding boxes y etiquetas en el video |

---

## Arquitectura de archivos

```
OmniFace-1/
├── main_gui.py          ← GUI completa (CustomTkinter) + loop de video + registro
├── app/
│   ├── recognition.py   ← Motor de reconocimiento (LBPH + detección + NMS)
│   ├── camera.py        ← Singleton VideoCamera — get_frame() → (frame, frame_id)
│   ├── database.py      ← SQLite: CRUD de identidades, muestras, logs
│   ├── alerts.py        ← AlertManager: contador de desconocidos + screenshots
│   ├── routes.py        ← NO USAR (leftover Flask/web)
│   └── __init__.py      ← NO USAR (leftover Flask/web)
├── data/
│   └── omniface.db      ← Base de datos SQLite (generada automáticamente)
├── assets/
│   └── logo.png
├── screenshots/         ← Capturas automáticas de desconocidos
├── CLAUDE.md            ← Knowledge Graph del proyecto (para Claude)
├── STACK.md             ← Este archivo
└── requirements.txt
```

---

## Base de datos (SQLite)

```sql
-- Personas registradas
identities (
    id          INTEGER PRIMARY KEY,
    name        TEXT NOT NULL,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)

-- Muestras faciales por persona (BLOBs JPEG 150x150 grises)
face_samples (
    id          INTEGER PRIMARY KEY,
    identity_id INTEGER NOT NULL,  -- FK → identities.id (CASCADE DELETE)
    face_data   BLOB NOT NULL
)

-- Historial de accesos detectados
access_log (
    id              INTEGER PRIMARY KEY,
    identity_id     INTEGER,
    identity_name   TEXT NOT NULL,
    confidence      REAL,
    status          TEXT NOT NULL,  -- 'known' | 'unknown'
    screenshot_path TEXT,
    timestamp       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
```

---

## Pipeline de reconocimiento

```
Frame de cámara
    │
    ▼
_detect_all_faces()
    ├── Cascade frontal          → caras 0°
    ├── Cascade perfil           → caras 90° derecha
    ├── Cascade perfil (flip)    → caras 90° izquierda
    └── NMS (IoU > 0.35)         → elimina duplicados solapados
    │
    ▼ (por cada cara detectada)
_prepare_face()
    ├── align_face()     → detecta ojos → warpAffine para horizontalizar
    ├── CLAHE            → normalización de iluminación
    └── resize(150,150)  → tamaño estándar
    │
    ▼
LBPH.predict()
    └── dist ≤ 95  → conocido, confianza = 100 - (dist × 40/95)
    └── dist > 95  → Desconocido
```

---

## Pipeline de entrenamiento (retrain)

```
DB face_samples (BLOBs)
    │
    ▼ (por cada muestra)
_prepare_face()          → alinear + CLAHE + resize(150,150)
    │
    ▼
_augment()               → [ original, bright(×1.25), dark(×0.75), flip ]
                                                → 4× el dataset
    │
    ▼
LBPH.train(faces, labels)
```

---

## GUI — Vistas implementadas

| Vista | Estado | Descripción |
|-------|--------|-------------|
| `monitoring` | ✅ | Video en vivo con bounding boxes y nombre/confianza |
| `registration` | ✅ | Registro multi-ángulo automático (6 pasos × 10 muestras) |
| `database` | ✅ | Lista de identidades con opción de eliminar |
| `logs` / Historial | ❌ | Botón en sidebar existe — vista NO implementada |

---

## Parámetros clave ajustables

| Parámetro | Valor actual | Archivo |
|-----------|-------------|---------|
| `MAX_DIST` | `95` | `recognition.py` |
| `FACE_SIZE` | `(150, 150)` | `recognition.py` |
| `PROCESS_EVERY_N` | `2` (1 de cada 2 frames) | `recognition.py` |
| `LOG_COOLDOWN` | `5` segundos | `recognition.py` |
| LBPH `radius` | `2` | `recognition.py` |
| Muestras por paso | `10` | `main_gui.py` |
| Pasos de registro | `6` | `main_gui.py` |
| Resolución cámara | `640 × 480` | `camera.py` |

---

## Herramientas de desarrollo

| Herramienta | Uso |
|-------------|-----|
| **Git** | Control de versiones |
| **GitHub** — `AngelMtz6/OmniFace` | Repositorio remoto, rama `main` |
| **VS Code / Cursor** | Editor |
| **Claude** | Asistente de vibe coding |
