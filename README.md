# OmniFace
Sistema inteligente de reconocimiento facial mediante visión artificial.

## Stack
- **Backend:** Python 3.11 + Flask
- **Reconocimiento:** face_recognition (dlib) + OpenCV
- **Base de datos:** SQLite
- **Frontend:** HTML / CSS / JS (sin frameworks)

## Instalación en Windows

### 1. Requisitos previos
```
# Instalar cmake (necesario para dlib)
pip install cmake

# Instalar dlib (puede tardar 2-3 minutos)
pip install dlib

# Instalar el resto de dependencias
pip install -r requirements.txt
```

> **Alternativa rápida:** si `dlib` falla, descarga el wheel precompilado desde  
> https://github.com/jloh02/dlib/releases  
> y ejecuta: `pip install dlib-19.24.x-cp311-win_amd64.whl`

### 2. Ejecutar
```
python run.py
```
Abre el navegador en **http://localhost:5000**

---

## Flujo de uso

1. **Registrar** — Ve a `/register`, escribe el nombre de la persona y captura 5+ fotos con la cámara web.
2. **Monitor** — Ve a `/` para ver el feed en vivo con reconocimiento en tiempo real.
3. **Panel** — Ve a `/admin` para ver estadísticas, historial de accesos y gestionar identidades.

---

## Estructura del proyecto
```
OmniFace/
├── app/
│   ├── __init__.py      # App factory Flask
│   ├── camera.py        # Captura de video (singleton thread-safe)
│   ├── database.py      # Operaciones SQLite
│   ├── recognition.py   # Motor de reconocimiento facial
│   ├── alerts.py        # Sistema de alertas
│   └── routes.py        # Endpoints y streaming
├── static/
│   ├── css/style.css
│   └── js/app.js
├── templates/
│   ├── base.html
│   ├── index.html       # Monitor en vivo
│   ├── register.html    # Registro de identidades
│   └── admin.html       # Panel de administración
├── data/                # SQLite DB (generado automáticamente)
├── screenshots/         # Capturas de desconocidos (generado automáticamente)
├── requirements.txt
└── run.py
```

---

## API endpoints

| Método | Ruta | Descripción |
|--------|------|-------------|
| GET | `/video_feed` | Stream MJPEG con reconocimiento |
| GET | `/capture_frame` | Captura un frame como base64 |
| POST | `/enroll` | Registrar nueva identidad |
| GET | `/identities` | Listar todas las identidades |
| DELETE | `/identity/<id>` | Eliminar identidad |
| GET | `/logs` | Historial de accesos |
| GET | `/stats` | Estadísticas generales |
| GET | `/events` | SSE para alertas en tiempo real |

---

## Equipo
Hackatec 2026 — OmniFace
