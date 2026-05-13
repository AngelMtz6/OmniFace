# OmniFace-1 — Stack Completo

> Documento de referencia para el equipo. Todo lo que se usa en el proyecto.

---

## Lenguaje

| Elemento | Detalle |
|----------|---------|
| **Python** | 3.14 |
| **Entorno virtual** | `.venv\` — activar con `.venv\Scripts\activate` |
| **Entry point** | `run.bat` ó `.venv\Scripts\python.exe main_gui.py` |
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
├── main_gui.py          ← GUI completa (CustomTkinter) + loop de video + registro
├── run.bat              ← Lanzador con venv correcto
├── app/
│   ├── recognition.py   ← Motor InsightFace (RetinaFace + ArcFace + retrain)
│   ├── camera.py        ← Singleton VideoCamera — get_frame() thread-safe
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

-- Muestras faciales por persona (BLOBs JPEG 224×224 color)
-- Formato nuevo: 224×224 BGR JPEG — compatible con RetinaFace en retrain
-- Formato antiguo (ignorado): 100×100 grises LBPH
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

## GUI — Vistas implementadas

| Vista | Estado | Descripción |
|-------|--------|-------------|
| `monitoring` | ✅ | Video en vivo con bounding boxes y nombre/confianza. Inferencia en hilo de fondo |
| `registration` | ✅ | Registro multi-ángulo automático (6 pasos × 10 muestras = 60 crops 224×224) |
| `database` | ✅ | Lista de identidades con opción de eliminar |
| `logs` / Historial | ❌ | Botón en sidebar existe — vista NO implementada |

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
| **Git** | Control de versiones |
| **GitHub** — `AngelMtz6/OmniFace` | Repositorio remoto, rama `main` |
| **VS Code** | Editor — `.vscode/launch.json` configurado para venv |
| **Claude** | Asistente de desarrollo |
| **nvidia-smi** | Verificar GPU disponible |
| **CUDA Toolkit 12.x** | Requerido para activar GPU (proporciona `cublasLt64_12.dll`) |
