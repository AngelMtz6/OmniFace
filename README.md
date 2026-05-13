# OmniFace
Sistema inteligente de reconocimiento facial en tiempo real con interfaz de escritorio y portal web.

## Stack
| Capa | Tecnología |
|------|------------|
| **Reconocimiento** | InsightFace — RetinaFace (detección) + ArcFace (embeddings 512-dim) |
| **Runtime DL** | ONNX Runtime — GPU (CUDA 12) o CPU automático |
| **GUI escritorio** | Python 3.14 + CustomTkinter + OpenCV |
| **Web** | Flask 3 + Bootstrap 5 (dark theme) |
| **Base de datos** | SQLite 3 |
| **Sincronización** | Git — push/pull de `data/omniface.db` |
| **HTTPS local** | ngrok (pyngrok) |

---

## Instalación

```bash
# 1. Clonar
git clone https://github.com/AngelMtz6/OmniFace
cd OmniFace-1

# 2. Crear entorno virtual
python -m venv .venv
.venv\Scripts\activate

# 3. Instalar dependencias
pip install -r requirements.txt
```

> **GPU (opcional):** Instalar CUDA Toolkit 12.x para activar el modo GPU.
> Sin CUDA el sistema corre en CPU automáticamente con el modelo compacto `buffalo_sc`.

---

## Ejecutar

### App de escritorio (monitor en vivo + registro)
```bash
run.bat
# o
.venv\Scripts\python.exe main_gui.py
```

### Portal web (registro facial, perfil, historial)
```bash
# HTTP — solo localhost
.venv\Scripts\python.exe run.py

# HTTPS con ngrok — permite cámara desde cualquier dispositivo
.venv\Scripts\python.exe run_ngrok.py
```
Abre la URL que imprime en consola (ej. `https://xxxx.ngrok-free.app`).

---

## Estructura del proyecto

```
OmniFace-1/
├── main_gui.py          # App de escritorio — monitor, registro, gestión
├── run.py               # Lanzador Flask (HTTP localhost:5000)
├── run_ngrok.py         # Lanzador Flask + túnel HTTPS ngrok
├── run.bat              # Lanzador escritorio con venv correcto
├── run_ngrok.bat        # Lanzador web con ngrok (doble clic)
│
├── app/
│   ├── __init__.py      # App factory Flask
│   ├── routes.py        # Rutas web (login, dashboard, perfil, historial, API)
│   ├── auth.py          # Autenticación — hashing, sesiones, reset de contraseña
│   ├── camera.py        # Singleton VideoCamera thread-safe
│   ├── database.py      # CRUD SQLite — identidades, muestras, logs, stats
│   ├── recognition.py   # Motor InsightFace — detección + embeddings + retrain
│   ├── alerts.py        # AlertManager — capturas de desconocidos
│   └── sync.py          # Sincronización Git — push_db(), pull_db_identities(),
│                        # start_detection_pusher() (push cada 10 s)
│
├── templates/           # Jinja2 — Bootstrap 5 dark
│   ├── base.html
│   ├── login.html
│   ├── register.html
│   ├── dashboard.html
│   ├── profile.html
│   ├── my_history.html
│   ├── register_face.html
│   └── forgot_password.html
│
├── static/
│   └── screenshots/     # Capturas automáticas de desconocidos
│
├── data/
│   └── omniface.db      # Base de datos SQLite (generada automáticamente)
│
├── STACK.md             # Referencia técnica detallada del stack
├── CLAUDE.md            # Knowledge graph del proyecto
└── requirements.txt
```

---

## Flujo de uso

### Registro vía web
1. Abre el portal web (`run_ngrok.py`).
2. Crea una cuenta en `/register` con CURP y datos personales.
3. En el dashboard aparece el banner **"Registrar rostro"** — haz clic.
4. La cámara captura 60 frames automáticamente y los envía al servidor.
5. El servidor guarda la identidad y hace **push a Git** para sincronizar.

### Monitor de escritorio
1. Ejecuta `run.bat`.
2. Pulsa **Sincronizar Nube** para importar identidades registradas vía web.
3. La app hace `pull_db_identities()` — solo importa identidades nuevas, no sobreescribe logs locales.
4. El reconocimiento corre en tiempo real; las detecciones se pushean a Git **cada 10 segundos** en lote.

### Portal web — historial
- El dashboard muestra estadísticas propias del usuario (detecciones totales, lugares visitados, hoy, última).
- "Mis últimas detecciones" muestra una entrada por cámara/lugar (sin duplicados).
- `/my-history` muestra el historial completo.

---

## Base de datos

```sql
accounts     -- Cuentas de usuario (web + desktop)
identities   -- Personas registradas con datos personales y account_id
face_samples -- BLOBs JPEG 224×224 por identidad (hasta 60 por persona)
access_log   -- Historial de detecciones con confianza, cámara y screenshot
cameras      -- Cámaras configuradas en la app de escritorio
```

---

## Sincronización Git

| Evento | Acción |
|--------|--------|
| Registro facial (web) | `push_db()` inmediato en hilo de fondo |
| Detecciones (desktop) | `push_db()` cada 10 s si hay nuevas (`start_detection_pusher`) |
| Eliminar identidad (desktop) | `push_db()` inmediato |
| Botón "Sincronizar Nube" (desktop) | `pull_db_identities()` — importa solo identidades nuevas del remoto |

---

## Parámetros clave

| Parámetro | Valor | Archivo |
|-----------|-------|---------|
| `COSINE_THRESHOLD` | `0.40` | `recognition.py` |
| `LOG_COOLDOWN` | `5 s` | `recognition.py` |
| `PROCESS_EVERY_N` | `1` (GPU) / `3` (CPU) | `recognition.py` |
| Modelo GPU | `buffalo_l` (ResNet50) | `recognition.py` |
| Modelo CPU | `buffalo_sc` (MobileFaceNet) | `recognition.py` |
| Push automático | cada `10 s` | `sync.py` |
| Muestras por registro | `60` (6 pasos × 10) | `main_gui.py` |

---

## Equipo
Hackatec 2026 — OmniFace
