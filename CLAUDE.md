# OmniFace-1 — Knowledge Graph (Desktop App)

> LEE ESTE ARCHIVO PRIMERO. Evita releer el código fuente innecesariamente.

---

## Identidad
- **Tipo:** App de escritorio — CustomTkinter (NO web)
- **Entry point:** `python main_gui.py`  ← único archivo para correr
- **run.py + routes.py:** son residuos de la versión web — NO se usan en desktop
- **Repo:** https://github.com/AngelMtz6/OmniFace  (rama `main`)
- **Directorio:** `C:\Users\angel\Desktop\Hackatec\OmniFace-1`
- **Venv:** `.venv\` (activar con `.venv\Scripts\activate`)

---

## Stack
| Capa | Tecnología |
|------|-----------|
| GUI | `customtkinter` 5.x + `tkinter` (Label para video) |
| Video | OpenCV CAP_DSHOW, flip horizontal, 640×480 |
| Reconocimiento | LBPH `radius=1, neighbors=8` + CLAHE + augmentation (bright/dark) |
| Detección | Haar frontal + Haar perfil (flip para perfil izquierdo) |
| Storage | SQLite — blobs JPEG 100×100 grises |
| System tray | `pystray` (minimizar a bandeja) |

---

## Árbol de archivos (solo los relevantes)

```
OmniFace-1/
├── main_gui.py          ← TODA la GUI: vistas, lógica de video, registro
├── run.py               ← NO USAR (web leftover)
├── app/
│   ├── camera.py        ← VideoCamera singleton, get_frame() → (frame, frame_id)
│   ├── recognition.py   ← RecognitionEngine: LBPH + CLAHE + augment + perfil
│   ├── database.py      ← SQLite: identities + face_samples (BLOBs) + access_log
│   ├── alerts.py        ← AlertManager: contador de desconocidos + screenshot
│   ├── routes.py        ← NO USAR (web leftover)
│   └── __init__.py      ← NO USAR (web leftover)
├── assets/logo.png
├── scratch/test_camera.py
└── requirements.txt     ← flask, opencv-contrib, numpy, Pillow, customtkinter, pystray
```

---

## Estructura de main_gui.py

```
OmniFaceApp(ctk.CTk)
├── __init__()
│   ├── VideoCamera(video_source=0)
│   ├── RecognitionEngine()
│   ├── Sidebar: botones Monitor / Registrar / Base de Datos / Historial
│   └── init_monitoring_view() + init_database_view() + init_registration_view()
│
├── Vistas (show_view switch)
│   ├── "monitoring"   → view_monitoring  (video_label, status_panel)
│   ├── "registration" → view_registration (video_reg_label, pasos)
│   ├── "database"     → view_database    (scrollable list de identidades)
│   └── "logs"         → ⚠ BOTÓN EN SIDEBAR PERO VISTA NO IMPLEMENTADA
│
├── update_video()  ← loop principal via self.after(20, ...)
│   ├── monitoring → engine.process_frame() → display_frame(video_label)
│   └── registration → handle_auto_registration(frame) → display_frame(video_reg_label)
│
├── handle_auto_registration(frame)   ← captura automática por pasos
│   ├── 6 pasos × 10 muestras = 60 capturas totales
│   ├── Pausa 3s entre pasos (_last_step_time)
│   ├── Detecta frontal Y perfil (pasos 5-6 usan profile_detector)
│   └── finish_registration() → save_identity() + retrain()
│
└── System Tray (pystray) en hilo daemon
```

---

## Estado de registro (variables en self)
```python
self.registration_steps   = [6 strings de instrucción]
self.samples_per_step     = 10
self.current_step         = 0   # 0=inactivo, 1-6=paso activo, 7=terminado
self.current_step_samples = 0   # muestras capturadas en el paso actual
self.captured_samples     = []  # lista de blobs JPEG
self.is_capturing_auto    = bool
self._last_step_time      = float (timestamp de fin de paso)
```

---

## Parámetros clave de recognition.py
| Variable | Valor | Efecto |
|----------|-------|--------|
| `LBPH_THRESH` | `85` (en código usa `MAX_DIST=100`) | Umbral de distancia |
| `PROCESS_EVERY_N` | `3` | Frames saltados |
| `LOG_COOLDOWN` | `5` seg | Entre logs |
| `FACE_SIZE` | `(100,100)` | Normalización |

Augmentation en retrain: por cada muestra → 3 variantes (original, bright, dark) → **3× el dataset**

---

## DB Schema
```sql
identities   (id, name, created_at)
face_samples (id, identity_id FK, face_data BLOB)   -- JPEG 100x100 gray
access_log   (id, identity_id, identity_name, confidence, status, screenshot_path, timestamp)
```

---

## Problemas conocidos
- `"logs"` view: botón en sidebar **no funciona** — falta `init_logs_view()` y case en `show_view()`
- `routes.py` y `__init__.py` en `app/` son web leftovers — no eliminar por si acaso
- Registration UI: solo texto, sin step bubbles, sin indicador visual de detección

---

## Estado al último commit
- [x] App de escritorio funcional con customtkinter
- [x] Registro multi-ángulo automático (6 pasos × 10 muestras)
- [x] LBPH + CLAHE + augmentation bright/dark
- [x] Detección frontal + perfil
- [x] System tray (pystray)
- [x] Gestión de identidades (DB view)
- [ ] Vista "Historial/Logs" — NO implementada
- [ ] UI de registro: sin step bubbles ni indicador visual de cara detectada
