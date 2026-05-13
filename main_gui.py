import sys, os, subprocess

# ── Auto-relanzar con el venv si insightface no está disponible ───────────────
_HERE        = os.path.dirname(os.path.abspath(__file__))
_VENV_PYTHON = os.path.join(_HERE, '.venv', 'Scripts', 'python.exe')

try:
    import insightface as _insightface_check  # noqa: F401
except ImportError:
    if os.path.exists(_VENV_PYTHON) and os.path.abspath(sys.executable) != os.path.abspath(_VENV_PYTHON):
        print(f"[OmniFace] Relanzando con venv: {_VENV_PYTHON}")
        result = subprocess.run([_VENV_PYTHON] + sys.argv)
        sys.exit(result.returncode)
    else:
        print("[OmniFace] ERROR: insightface no instalado en este entorno.")
        print(f"[OmniFace] Ejecuta: {_VENV_PYTHON} main_gui.py")
        sys.exit(1)
# ─────────────────────────────────────────────────────────────────────────────

import tkinter as tk
from tkinter import messagebox
# pyrefly: ignore [missing-import]
import customtkinter as ctk
import cv2
from PIL import Image, ImageTk
import threading
import time
import os
import sys
import pystray
from pystray import MenuItem as item
from app.camera import VideoCamera
from app.recognition import RecognitionEngine
from app.database import init_db, get_all_identities, delete_identity, get_access_logs, get_stats

# Configuración estética
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class OmniFaceApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("OmniFace - Sistema de Reconocimiento Facial")
        self.geometry("1100x700")
        
        # Asegurar que la DB esté lista
        init_db()

        # Iniciar push periódico de detecciones cada 10 s
        from app.sync import start_detection_pusher
        start_detection_pusher()
        
        # Inicializar componentes
        self.camera_index = 0
        self.camera_label  = f"Cámara {self.camera_index}"  # nombre para log_access
        self.camera = VideoCamera(video_source=self.camera_index)
        self.engine = RecognitionEngine()
        
        # Variables de estado
        self.last_frame_id = -1
        self.current_view = "monitoring"
        self.is_minimized = False
        
        # Estado de Verificación Liveness (3D simulado)
        self.is_verifying = False
        self.verifying_name = ""
        self.verify_start_time = 0
        self.current_challenge = "" # "Frente", "Perfil"
        self.verify_step = 0
        self.verify_timeout = 20 # segundos
        self.challenges = ["Perfil", "Frente"] # Secuencia simple para simular 3D
        
        # Estado de Registro
        self.registration_name = ""
        self.registration_account_id = None   # ID de cuenta seleccionada para registrar
        self.accounts = []                     # Lista cacheada de cuentas (para finish_registration)
        self.current_step = 0
        self.captured_samples = []
        self.samples_per_step = 10  # 10 × 6 pasos = 60 muestras (ArcFace no necesita más)
        self.current_step_samples = 0
        self.is_capturing_auto = False
        self.registration_steps = [
            "Ponte de frente, muy de cerca",
            "Ahora aléjate un poco",
            "Mira ligeramente hacia arriba",
            "Mira ligeramente hacia abajo",
            "Gira la cabeza hacia un lado",
            "Gira la cabeza hacia el otro lado"
        ]
        
        # Configurar Grid
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # ── Sidebar ──
        self.sidebar = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        
        self.logo_label = ctk.CTkLabel(self.sidebar, text="OMNIFACE", font=ctk.CTkFont(size=20, weight="bold"))
        self.logo_label.pack(pady=20, padx=10)

        self.btn_monitor = ctk.CTkButton(self.sidebar, text="Monitoreo", command=lambda: self.show_view("monitoring"))
        self.btn_monitor.pack(pady=10, padx=20)

        self.btn_register = ctk.CTkButton(self.sidebar, text="Registrar Usuario", command=lambda: self.show_view("registration"))
        self.btn_register.pack(pady=10, padx=20)

        self.btn_db = ctk.CTkButton(self.sidebar, text="Base de Datos", command=lambda: self.show_view("database"))
        self.btn_db.pack(pady=10, padx=20)

        self.btn_logs = ctk.CTkButton(self.sidebar, text="Historial", command=lambda: self.show_view("logs"))
        self.btn_logs.pack(pady=10, padx=20)

        self.btn_search = ctk.CTkButton(self.sidebar, text="Búsqueda", command=lambda: self.show_view("search"))
        self.btn_search.pack(pady=10, padx=20)

        self.btn_sync = ctk.CTkButton(self.sidebar, text="Sincronizar Nube", fg_color="#1f538d", hover_color="#14375e", command=self.sync_cloud)
        self.btn_sync.pack(pady=10, padx=20)

        self.btn_exit = ctk.CTkButton(self.sidebar, text="Salir del Sistema", fg_color="#441111", hover_color="#662222", command=self.quit)
        self.btn_exit.pack(side="bottom", pady=(10, 5), padx=20)

        self.sidebar_footer = ctk.CTkLabel(self.sidebar, text="Hackatec 2026", font=ctk.CTkFont(size=10))
        self.sidebar_footer.pack(side="bottom", pady=(5, 10))

        # ── Main Content Area ──
        self.main_content = ctk.CTkFrame(self, corner_radius=15, fg_color="transparent")
        self.main_content.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)

        # Vistas
        self.init_monitoring_view()
        self.init_database_view()
        self.init_registration_view()
        self.init_logs_view()
        self.init_search_view()

        self.show_view("monitoring")

        # ── System Tray ──
        self.setup_tray()

        # Estado del procesamiento en fondo (para cámara fluida)
        self._processing      = False
        self._display_frame   = None   # último frame ya procesado (con bboxes)

        # Iniciar loop de video
        self.update_video()
        
        # Protocolo de cierre
        self.protocol("WM_DELETE_WINDOW", self.minimize_to_tray)

    # ── Vistas ──

    def init_monitoring_view(self):
        self.view_monitoring = ctk.CTkFrame(self.main_content, fg_color="transparent")

        # ── Barra de controles (se reserva primero → nunca queda tapada) ──────
        self.status_panel = ctk.CTkFrame(self.view_monitoring, height=44, corner_radius=8)
        self.status_panel.pack(side="bottom", fill="x", padx=10, pady=(4, 10))
        self.status_panel.pack_propagate(False)

        self.status_text = ctk.CTkLabel(
            self.status_panel, text="⬤  Sistema activo",
            text_color="#50CD64", font=ctk.CTkFont(weight="bold"))
        self.status_text.pack(side="left", padx=14)

        self.btn_switch_cam = ctk.CTkButton(
            self.status_panel, text="Cambiar cámara", width=130,
            command=self.switch_camera)
        self.btn_switch_cam.pack(side="right", padx=8)

        self.btn_clear_log = ctk.CTkButton(
            self.status_panel, text="Limpiar log", width=110,
            fg_color="#2a2a3a", hover_color="#3a3a4a",
            command=self._clear_monitor_log)
        self.btn_clear_log.pack(side="right", padx=4)

        # ── Contenido principal: video + log lateral ──────────────────────────
        content = ctk.CTkFrame(self.view_monitoring, fg_color="transparent")
        content.pack(expand=True, fill="both", padx=10, pady=(10, 4))
        content.grid_columnconfigure(0, weight=3)
        content.grid_columnconfigure(1, weight=1, minsize=220)
        content.grid_rowconfigure(0, weight=1)

        # Feed de vídeo
        self.video_label = tk.Label(content, bg="#1a1a1a")
        self.video_label.grid(row=0, column=0, sticky="nsew", padx=(0, 6))

        # Panel log en vivo
        log_panel = ctk.CTkFrame(content, fg_color="#12121e", corner_radius=10)
        log_panel.grid(row=0, column=1, sticky="nsew")
        log_panel.grid_rowconfigure(1, weight=1)
        log_panel.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            log_panel, text="Detecciones recientes",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#8888aa"
        ).grid(row=0, column=0, pady=(10, 4), padx=10, sticky="w")

        self.monitor_log_frame = ctk.CTkScrollableFrame(
            log_panel, fg_color="transparent", corner_radius=0)
        self.monitor_log_frame.grid(row=1, column=0, sticky="nsew", padx=4, pady=(0, 6))

        # Actualizar el log lateral cada 4 s
        self._schedule_monitor_log_refresh()

    def switch_camera(self):
        try:
            self.camera_index = (self.camera_index + 1) % 3
            self.camera_label = f"Cámara {self.camera_index}"
            self.camera.release()
            self.camera._initialized = False
            self.camera.__init__(video_source=self.camera_index)
            time.sleep(0.5)
            if not self.camera.cap.isOpened():
                messagebox.showwarning("Cámara", f"No se pudo abrir la cámara #{self.camera_index}.\n\nAsegúrate de que no esté bloqueada por otra aplicación.")
            else:
                messagebox.showinfo("Cámara", f"Cambiando a fuente de video #{self.camera_index}")
        except Exception as e:
            messagebox.showerror("Error", f"Error al cambiar cámara: {e}")

    # ── Log lateral de monitoreo ──────────────────────────────────────────────

    def _schedule_monitor_log_refresh(self):
        """Programa la actualización del log lateral cada 4 s."""
        self._refresh_monitor_log()
        self.after(4000, self._schedule_monitor_log_refresh)

    def _refresh_monitor_log(self):
        """Rellena el panel lateral con las últimas 20 detecciones."""
        if not hasattr(self, 'monitor_log_frame'):
            return
        from app.database import get_access_logs
        for w in self.monitor_log_frame.winfo_children():
            w.destroy()
        logs = get_access_logs(limit=20)
        if not logs:
            ctk.CTkLabel(self.monitor_log_frame, text="Sin detecciones aún",
                         text_color="#444466", font=ctk.CTkFont(size=11)).pack(pady=20)
            return
        for log in logs:
            known  = log['status'] == 'known'
            color  = "#1a2e1a" if known else "#2e1a1a"
            tcolor = "#50CD64" if known else "#FF5555"
            ts     = (log['timestamp'] or "")[:16].replace("T", " ")
            conf   = log['confidence']
            conf_s = f"  {conf:.0f}%" if conf is not None else ""

            card = ctk.CTkFrame(self.monitor_log_frame, fg_color=color,
                                corner_radius=6, height=52)
            card.pack(fill="x", pady=2, padx=2)
            card.pack_propagate(False)

            ctk.CTkLabel(card,
                         text="✓" if known else "?",
                         text_color=tcolor,
                         font=ctk.CTkFont(size=14, weight="bold"),
                         width=24).place(x=6, y=8)
            ctk.CTkLabel(card,
                         text=log['identity_name'],
                         font=ctk.CTkFont(size=11, weight="bold"),
                         text_color="#dddddd",
                         anchor="w").place(x=30, y=6)
            ctk.CTkLabel(card,
                         text=f"{ts}{conf_s}",
                         font=ctk.CTkFont(size=10),
                         text_color="#777799",
                         anchor="w").place(x=30, y=26)

    def _clear_monitor_log(self):
        if not hasattr(self, 'monitor_log_frame'):
            return
        for w in self.monitor_log_frame.winfo_children():
            w.destroy()
        ctk.CTkLabel(self.monitor_log_frame, text="Log limpiado",
                     text_color="#444466", font=ctk.CTkFont(size=11)).pack(pady=20)

    def init_database_view(self):
        self.view_database = ctk.CTkFrame(self.main_content, fg_color="transparent")
        
        self.db_title = ctk.CTkLabel(self.view_database, text="Gestión de Identidades", font=ctk.CTkFont(size=24, weight="bold"))
        self.db_title.pack(pady=20)
        
        self.identities_list = ctk.CTkScrollableFrame(self.view_database, width=600, height=400)
        self.identities_list.pack(pady=10, padx=20, expand=True, fill="both")
        
        self.btn_refresh = ctk.CTkButton(self.view_database, text="Actualizar Lista", command=self.refresh_identities)
        self.btn_refresh.pack(pady=10)

    def init_registration_view(self):
        self.view_registration = ctk.CTkFrame(self.main_content, fg_color="transparent")

        self.reg_title = ctk.CTkLabel(self.view_registration, text="Nuevo Registro",
                                      font=ctk.CTkFont(size=22, weight="bold"))
        self.reg_title.pack(pady=(15, 5))

        # ── Cuenta ──
        name_row = ctk.CTkFrame(self.view_registration, fg_color="transparent")
        name_row.pack(pady=5)
        
        from app.auth import get_all_accounts
        self.accounts = get_all_accounts()
        self.account_options = [f"{a['id']} - {a['nombre']} {a['ap_paterno']}" for a in self.accounts]
        if not self.account_options:
            self.account_options = ["(No hay cuentas disponibles)"]
            
        self.account_combo = ctk.CTkComboBox(name_row, values=self.account_options, width=260)
        self.account_combo.pack(side="left", padx=(0, 8))
        self.btn_capture = ctk.CTkButton(name_row, text="Iniciar", width=100,
                                         command=self.handle_registration_click)
        self.btn_capture.pack(side="left")

        # ── Step bubbles (6 círculos de pasos) ──
        self.step_bubbles_frame = ctk.CTkFrame(self.view_registration, fg_color="transparent")
        self.step_bubbles_frame.pack(pady=(8, 2))
        self._step_icons  = ["😐", "⬅", "➡", "⬆", "⬇", "😊"]
        self._step_labels_short = ["Frente", "Izq", "Der", "Arriba", "Abajo", "Sonríe"]
        self._bubble_widgets = []
        for i in range(6):
            col = ctk.CTkFrame(self.step_bubbles_frame, fg_color="transparent")
            col.pack(side="left", padx=6)
            circle = ctk.CTkLabel(col, text=self._step_icons[i],
                                  width=46, height=46,
                                  corner_radius=23,
                                  fg_color="#2a2a3a",
                                  font=ctk.CTkFont(size=18))
            circle.pack()
            lbl = ctk.CTkLabel(col, text=self._step_labels_short[i],
                               font=ctk.CTkFont(size=10), text_color="#666666")
            lbl.pack()
            self._bubble_widgets.append((circle, lbl))

        # ── Instrucción actual ──
        self.instruction_label = ctk.CTkLabel(self.view_registration,
                                              text="Selecciona una cuenta y presiona Iniciar",
                                              font=ctk.CTkFont(size=14), text_color="#AAAAAA")
        self.instruction_label.pack(pady=(6, 2))

        # ── Indicador de detección ──
        self.face_detect_label = ctk.CTkLabel(self.view_registration, text="⬤  Buscando rostro…",
                                              font=ctk.CTkFont(size=12), text_color="#555555")
        self.face_detect_label.pack(pady=(0, 4))

        # ── Video ──
        self.video_reg_frame = ctk.CTkFrame(self.view_registration, corner_radius=10,
                                            border_width=3, border_color="#2a2a3a")
        self.video_reg_frame.pack(pady=4)
        self.video_reg_label = tk.Label(self.video_reg_frame, bg="#1a1a1a", width=400, height=280)
        self.video_reg_label.pack(padx=3, pady=3)

        # ── Barra de progreso del paso actual ──
        self.step_progress = ctk.CTkProgressBar(self.view_registration, width=400, height=8)
        self.step_progress.set(0)
        self.step_progress.pack(pady=(4, 2))

        self.progress_label = ctk.CTkLabel(self.view_registration, text="Paso 0/6  |  Muestras: 0/60",
                                           font=ctk.CTkFont(size=11), text_color="#888888")
        self.progress_label.pack()
        # total = 10 × 6 = 60

    # ── Vista Búsqueda especializada ──────────────────────────────────────────

    def init_search_view(self):
        self.view_search = ctk.CTkFrame(self.main_content, fg_color="transparent")

        # ── Título ──
        ctk.CTkLabel(self.view_search, text="Búsqueda especializada",
                     font=ctk.CTkFont(size=22, weight="bold")).pack(pady=(16, 4))
        ctk.CTkLabel(self.view_search,
                     text="Selecciona una identidad registrada para ver todo su historial de detecciones.",
                     font=ctk.CTkFont(size=12), text_color="#888888").pack()

        # ── Barra de búsqueda ──
        bar = ctk.CTkFrame(self.view_search, fg_color="#1a1a2e", corner_radius=10)
        bar.pack(fill="x", padx=20, pady=14)

        ctk.CTkLabel(bar, text="Identidad:", font=ctk.CTkFont(size=13)).pack(side="left", padx=(14, 6), pady=12)

        self.search_combo = ctk.CTkComboBox(bar, values=["Cargando..."], width=280,
                                            state="readonly")
        self.search_combo.pack(side="left", padx=6, pady=12)

        ctk.CTkLabel(bar, text="Estado:", font=ctk.CTkFont(size=13)).pack(side="left", padx=(16, 6))
        self.search_status_var = tk.StringVar(value="Todos")
        self.search_status_menu = ctk.CTkOptionMenu(
            bar, values=["Todos", "Conocido", "Desconocido"],
            variable=self.search_status_var, width=130)
        self.search_status_menu.pack(side="left", padx=6, pady=12)

        ctk.CTkButton(bar, text="Buscar", width=90,
                      command=self._run_search).pack(side="left", padx=10, pady=12)
        ctk.CTkButton(bar, text="Limpiar", width=80,
                      fg_color="#2a2a3a", hover_color="#3a3a4a",
                      command=self._clear_search).pack(side="left", padx=4)

        # ── Contador de resultados ──
        self.search_count_label = ctk.CTkLabel(
            self.view_search, text="", font=ctk.CTkFont(size=11), text_color="#666699")
        self.search_count_label.pack(anchor="w", padx=24)

        # ── Cabecera de tabla ──
        hdr = ctk.CTkFrame(self.view_search, fg_color="#2a2a3a", height=30, corner_radius=0)
        hdr.pack(fill="x", padx=20)
        hdr.pack_propagate(False)
        for txt, w in [("#", 40), ("Fecha / Hora", 160), ("Estado", 110),
                       ("Confianza", 90), ("Cámara", 160), ("Foto", 60)]:
            ctk.CTkLabel(hdr, text=txt, width=w,
                         font=ctk.CTkFont(size=12, weight="bold"),
                         anchor="w").pack(side="left", padx=6)

        # ── Lista de resultados ──
        self.search_results_frame = ctk.CTkScrollableFrame(
            self.view_search, fg_color="#111120", corner_radius=0)
        self.search_results_frame.pack(expand=True, fill="both", padx=20, pady=(0, 14))

    def _populate_search_combo(self):
        """Carga las identidades disponibles en el combo."""
        identities = get_all_identities()
        names = [f"{i['id']} — {i['name']}" for i in identities]
        if not names:
            names = ["(Sin identidades registradas)"]
        self.search_combo.configure(values=names)
        self.search_combo.set(names[0])

    def _run_search(self):
        """Ejecuta la búsqueda y muestra resultados."""
        from app.database import get_access_logs

        # Limpiar resultados anteriores
        for w in self.search_results_frame.winfo_children():
            w.destroy()

        selected = self.search_combo.get()
        if not selected or "Sin identidades" in selected:
            self.search_count_label.configure(text="No hay identidades para buscar.")
            return

        try:
            identity_id = int(selected.split(" — ")[0])
        except (ValueError, IndexError):
            self.search_count_label.configure(text="Selección inválida.")
            return

        status_filter = self.search_status_var.get()

        # Obtener todos los logs de esta identidad
        all_logs = get_access_logs(limit=500, identity_id=identity_id)

        # Aplicar filtro de estado
        if status_filter == "Conocido":
            all_logs = [l for l in all_logs if l['status'] == 'known']
        elif status_filter == "Desconocido":
            all_logs = [l for l in all_logs if l['status'] == 'unknown']

        total = len(all_logs)
        nombre = selected.split(" — ", 1)[1] if " — " in selected else selected
        self.search_count_label.configure(
            text=f"  {total} resultado(s) para '{nombre}' — filtro: {status_filter}")

        if not all_logs:
            ctk.CTkLabel(self.search_results_frame,
                         text="Sin resultados con los filtros aplicados.",
                         text_color="#555577",
                         font=ctk.CTkFont(size=13)).pack(pady=30)
            return

        for idx, log in enumerate(all_logs):
            known   = log['status'] == 'known'
            bg      = "#181828" if idx % 2 == 0 else "#141422"
            tcolor  = "#50CD64" if known else "#FF5555"
            ts      = (log['timestamp'] or "")[:16].replace("T", " ")
            conf    = log['confidence']
            conf_s  = f"{conf:.1f}%" if conf is not None else "—"
            status_s = "Conocido" if known else "Desconocido"

            row = ctk.CTkFrame(self.search_results_frame, fg_color=bg,
                               corner_radius=0, height=34)
            row.pack(fill="x", pady=1)
            row.pack_propagate(False)

            ctk.CTkLabel(row, text=str(idx + 1), width=40,
                         font=ctk.CTkFont(size=11), text_color="#555577",
                         anchor="e").pack(side="left", padx=6)
            ctk.CTkLabel(row, text=ts, width=160,
                         font=ctk.CTkFont(size=11), anchor="w").pack(side="left", padx=6)
            ctk.CTkLabel(row, text=status_s, width=110,
                         text_color=tcolor, font=ctk.CTkFont(size=11, weight="bold"),
                         anchor="w").pack(side="left", padx=6)
            ctk.CTkLabel(row, text=conf_s, width=90,
                         font=ctk.CTkFont(size=11), anchor="w").pack(side="left", padx=6)
            ctk.CTkLabel(row, text=log.get('camera_name') or "—", width=160,
                         font=ctk.CTkFont(size=11), text_color="#8888aa",
                         anchor="w").pack(side="left", padx=6)

            if log.get('screenshot_path') and os.path.exists(log['screenshot_path']):
                ctk.CTkButton(row, text="Ver", width=52,
                              font=ctk.CTkFont(size=10),
                              command=lambda p=log['screenshot_path']: self.view_screenshot(p)
                              ).pack(side="left", padx=4)
            else:
                ctk.CTkLabel(row, text="—", width=60,
                             font=ctk.CTkFont(size=11),
                             text_color="#333355").pack(side="left", padx=6)

    def _clear_search(self):
        for w in self.search_results_frame.winfo_children():
            w.destroy()
        self.search_count_label.configure(text="")

    def sync_cloud(self):
        """
        Importa solo las identidades/muestras nuevas del DB remoto (Git).
        NO reemplaza el DB completo — los logs, cámaras y cuentas locales
        se conservan intactos.
        """
        from app.sync import pull_db_identities

        def _bg_sync():
            try:
                self.after(0, lambda: self.btn_sync.configure(
                    state="disabled", text="⏳ Sincronizando…"))

                ok, msg = pull_db_identities()

                if ok:
                    # Reentrenar con los datos nuevos
                    self.after(0, lambda: self.btn_sync.configure(
                        text="⏳ Entrenando…"))
                    self.engine.retrain()

                    def _on_done():
                        # Refrescar vistas si corresponde
                        if self.current_view == "database":
                            self.refresh_identities()
                        messagebox.showinfo(
                            "Sincronización completada",
                            f"✓ {msg}\n\n"
                            "El motor ya reconoce a los nuevos usuarios."
                        )
                    self.after(0, _on_done)
                else:
                    self.after(0, lambda: messagebox.showerror(
                        "Error de sincronización", msg))

            except Exception as e:
                self.after(0, lambda: messagebox.showerror(
                    "Error inesperado", str(e)))
            finally:
                self.after(0, lambda: self.btn_sync.configure(
                    state="normal", text="Sincronizar Nube"))

        threading.Thread(target=_bg_sync, daemon=True).start()

    def show_view(self, view_name):
        self.view_monitoring.pack_forget()
        self.view_database.pack_forget()
        self.view_registration.pack_forget()
        self.view_logs.pack_forget()
        self.view_search.pack_forget()

        if view_name == "monitoring":
            self.view_monitoring.pack(expand=True, fill="both")
        elif view_name == "database":
            self.view_database.pack(expand=True, fill="both")
            self.refresh_identities()
        elif view_name == "logs":
            self.view_logs.pack(expand=True, fill="both")
            self.refresh_logs()
        elif view_name == "registration":
            self.reset_registration_state()
            self.view_registration.pack(expand=True, fill="both")
        elif view_name == "search":
            self.view_search.pack(expand=True, fill="both")
            self._populate_search_combo()

        self.current_view = view_name

    def refresh_identities(self):
        # Limpiar lista
        for widget in self.identities_list.winfo_children():
            widget.destroy()
            
        identities = get_all_identities()
        for idx, identity in enumerate(identities):
            row = ctk.CTkFrame(self.identities_list)
            row.pack(fill="x", pady=5, padx=5)
            
            ctk.CTkLabel(row, text=f"ID: {identity['id']}").pack(side="left", padx=10)
            ctk.CTkLabel(row, text=f"Nombre: {identity['name']}", font=ctk.CTkFont(weight="bold")).pack(side="left", padx=10)
            
            btn_del = ctk.CTkButton(row, text="Eliminar", width=80, fg_color="#CC3333", hover_color="#AA2222", 
                                   command=lambda i=identity['id']: self.remove_identity(i))
            btn_del.pack(side="right", padx=10)

    def remove_identity(self, identity_id):
        if messagebox.askyesno("Confirmar", "¿Eliminar esta identidad y sus muestras?"):
            from app.sync import push_db
            delete_identity(identity_id)
            self.refresh_identities()
            threading.Thread(target=self.engine.retrain, daemon=True).start()
            push_db("identidad eliminada")

    # ── Vista de Historial ──

    def init_logs_view(self):
        self.view_logs = ctk.CTkFrame(self.main_content, fg_color="transparent")
        
        self.logs_title = ctk.CTkLabel(self.view_logs, text="Historial de Accesos", font=ctk.CTkFont(size=24, weight="bold"))
        self.logs_title.pack(pady=(0, 20))
        
        # Header de tabla
        header = ctk.CTkFrame(self.view_logs, fg_color="#2a2a3a", height=30)
        header.pack(fill="x", padx=10, pady=(0, 5))
        
        ctk.CTkLabel(header, text="Fecha/Hora", width=150, font=ctk.CTkFont(size=12, weight="bold")).pack(side="left", padx=10)
        ctk.CTkLabel(header, text="Nombre", width=150, font=ctk.CTkFont(size=12, weight="bold")).pack(side="left", padx=10)
        ctk.CTkLabel(header, text="Estado", width=100, font=ctk.CTkFont(size=12, weight="bold")).pack(side="left", padx=10)
        ctk.CTkLabel(header, text="Confianza", width=80, font=ctk.CTkFont(size=12, weight="bold")).pack(side="left", padx=10)

        self.logs_list = ctk.CTkScrollableFrame(self.view_logs, fg_color="#1a1a2a")
        self.logs_list.pack(expand=True, fill="both", padx=10, pady=10)

    def refresh_logs(self):
        from app.database import get_access_logs
        
        # Limpiar lista
        for widget in self.logs_list.winfo_children():
            widget.destroy()
            
        logs = get_access_logs(limit=50)
        for log in logs:
            row = ctk.CTkFrame(self.logs_list, fg_color="#232333" if log['status'] == 'known' else "#331a1a")
            row.pack(fill="x", pady=2, padx=5)
            
            # Formatear timestamp (YYYY-MM-DD HH:MM:SS → DD/MM HH:MM)
            ts = log['timestamp'] or ""
            try:
                ts = ts[:16].replace("T", " ")
            except Exception:
                pass

            conf_val = log['confidence']
            conf_str = f"{conf_val:.1f}%" if conf_val is not None else "—"

            ctk.CTkLabel(row, text=ts, width=150, font=ctk.CTkFont(size=11)).pack(side="left", padx=10)
            ctk.CTkLabel(row, text=log['identity_name'], width=150, font=ctk.CTkFont(weight="bold")).pack(side="left", padx=10)

            status_color = "#50CD64" if log['status'] == 'known' else "#FF5555"
            status_text  = "CONOCIDO" if log['status'] == 'known' else "DESCONOCIDO"
            ctk.CTkLabel(row, text=status_text, width=100, text_color=status_color).pack(side="left", padx=10)

            ctk.CTkLabel(row, text=conf_str, width=80).pack(side="left", padx=10)
            
            if log['screenshot_path'] and os.path.exists(log['screenshot_path']):
                btn_view = ctk.CTkButton(row, text="Ver Foto", width=80, font=ctk.CTkFont(size=11),
                                        command=lambda p=log['screenshot_path']: self.view_screenshot(p))
                btn_view.pack(side="right", padx=10)

    def view_screenshot(self, path):
        try:
            # Abrir imagen con el visor predeterminado del sistema
            if sys.platform == "win32":
                os.startfile(path)
            elif sys.platform == "darwin":
                subprocess.run(["open", path])
            else:
                subprocess.run(["xdg-open", path])
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo abrir la imagen: {e}")

    # ── Lógica de Registro ──

    def reset_registration_state(self):
        self.current_step = 0
        self.current_step_samples = 0
        self.captured_samples = []
        self.registration_account_id = None
        self.is_capturing_auto = False
        if hasattr(self, 'account_combo'):
            from app.auth import get_all_accounts
            self.accounts = get_all_accounts()
            self.account_options = [f"{a['id']} - {a['nombre']} {a['ap_paterno']}" for a in self.accounts]
            if not self.account_options:
                self.account_options = ["(No hay cuentas disponibles)"]
            self.account_combo.configure(values=self.account_options, state="normal")
            self.account_combo.set(self.account_options[0])
            
            self.instruction_label.configure(text="Selecciona una cuenta y presiona Iniciar", text_color="#AAAAAA")
            self.btn_capture.configure(text="Iniciar", fg_color=['#3B8ED0', '#1F538D'], state="normal")
            self.progress_label.configure(text="Paso 0/6  |  Muestras: 0/60")
            self.face_detect_label.configure(text="⬤  Buscando rostro…", text_color="#555555")
            self.step_progress.set(0)
            self.video_reg_frame.configure(border_color="#2a2a3a")
            # Resetear bubbles
            for i, (circle, lbl) in enumerate(self._bubble_widgets):
                circle.configure(fg_color="#2a2a3a", text_color="white")
                lbl.configure(text_color="#666666")

    def handle_registration_click(self):
        if not self.is_capturing_auto:
            selected = self.account_combo.get()
            if not selected or selected == "(No hay cuentas disponibles)":
                messagebox.showwarning("Atención", "Por favor selecciona una cuenta")
                return
            
            try:
                acc_id_str = selected.split(" - ")[0]
                self.registration_account_id = int(acc_id_str)
            except Exception:
                messagebox.showwarning("Atención", "Selección de cuenta inválida")
                return
                
            self.account_combo.configure(state="disabled")
            self.btn_capture.configure(text="Registrando…", state="disabled")
            self.is_capturing_auto = True
            self.current_step = 1
            self.update_registration_ui()

    def _update_step_bubbles(self, active_step):
        """Pinta los círculos de pasos: gris=pendiente, azul=activo, verde=completado."""
        for i, (circle, lbl) in enumerate(self._bubble_widgets):
            step_num = i + 1
            if step_num < active_step:
                circle.configure(fg_color="#1a6b3a", text_color="white")  # verde completado
                lbl.configure(text_color="#50CD64")
            elif step_num == active_step:
                circle.configure(fg_color="#1F538D", text_color="white")  # azul activo
                lbl.configure(text_color="#5AABFF")
            else:
                circle.configure(fg_color="#2a2a3a", text_color="white")  # gris pendiente
                lbl.configure(text_color="#666666")

    def update_registration_ui(self):
        if self.current_step <= len(self.registration_steps):
            instruction = self.registration_steps[self.current_step - 1]
            self.instruction_label.configure(text=instruction, text_color="#50CD64")
            total = len(self.registration_steps) * self.samples_per_step
            done  = (self.current_step - 1) * self.samples_per_step + self.current_step_samples
            self.progress_label.configure(
                text=f"Paso {self.current_step}/6  |  Muestras: {done}/{total}")
            self.step_progress.set(self.current_step_samples / self.samples_per_step)
            self._update_step_bubbles(self.current_step)
        else:
            self.finish_registration()

    def handle_auto_registration(self, frame):
        """
        Captura automática usando InsightFace (RetinaFace).
        Guarda crops 112×112 color — compatibles con ArcFace en retrain().
        """
        if not self.is_capturing_auto:
            return

        # Pausa breve entre poses para que el usuario se reubique
        if hasattr(self, '_last_step_time') and time.time() - self._last_step_time < 3.0:
            self.instruction_label.configure(text="¡Prepárate! Siguiente pose...", text_color="#FFA500")
            return

        # ── Detección con InsightFace (maneja frente + perfil automáticamente) ──
        faces = self.engine.detect_faces(frame)

        if faces:
            # Cara detectada — tomar la más grande (mayor área de bbox)
            face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))

            self.face_detect_label.configure(text="⬤  ¡Rostro detectado! Capturando…",
                                             text_color="#50CD64")
            self.video_reg_frame.configure(border_color="#50CD64")

            # Extraer crop con 50% de padding — RetinaFace necesita contexto
            # para detectar el rostro durante retrain()
            fh, fw = frame.shape[:2]
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
            bw, bh = x2 - x1, y2 - y1
            px, py = int(bw * 0.5), int(bh * 0.5)
            x1p = max(0, x1 - px);  y1p = max(0, y1 - py)
            x2p = min(fw, x2 + px); y2p = min(fh, y2 + py)

            crop = frame[y1p:y2p, x1p:x2p]
            if crop.size == 0:
                return

            # 224×224 color — retrain() puede detectar el rostro con RetinaFace
            crop_224 = cv2.resize(crop, (224, 224))
            _, buf   = cv2.imencode('.jpg', crop_224, [cv2.IMWRITE_JPEG_QUALITY, 92])

            self.captured_samples.append(buf.tobytes())
            self.current_step_samples += 1

            if self.current_step_samples >= self.samples_per_step:
                self._update_step_bubbles(self.current_step)
                self.current_step += 1
                self.current_step_samples = 0
                self._last_step_time = time.time()
                if self.current_step <= len(self.registration_steps):
                    self.face_detect_label.configure(
                        text=f"✓  Paso {self.current_step - 1}/6 completo — prepárate para el siguiente",
                        text_color="#FFA500")
                    self.video_reg_frame.configure(border_color="#FFA500")
                    self.step_progress.set(0)

            self.update_registration_ui()
        else:
            self.face_detect_label.configure(text="⬤  Buscando rostro…", text_color="#FF5555")
            self.video_reg_frame.configure(border_color="#552222")

    def finish_registration(self):
        from app.database import save_identity_full

        n_samples = len(self.captured_samples)
        
        acc = next((a for a in self.accounts if a['id'] == self.registration_account_id), None)
        if not acc:
            messagebox.showerror("Error", "No se encontró la cuenta para guardar la identidad.")
            self.reset_registration_state()
            return
            
        name = f"{acc['nombre']} {acc['ap_paterno']}".strip()
        
        save_identity_full(
            nombre=acc['nombre'],
            ap_paterno=acc['ap_paterno'],
            ap_materno=acc['ap_materno'],
            curp=acc['curp'],
            fecha_nac=acc['fecha_nac'],
            correo=acc['correo'],
            face_blobs=self.captured_samples,
            account_id=acc['id']
        )

        # Retrain en hilo de fondo para no congelar la GUI
        self.instruction_label.configure(
            text="⏳ Entrenando modelo… por favor espera", text_color="#FFA500")
        self.btn_capture.configure(state="disabled")
        self.is_capturing_auto = False

        def _do_retrain():
            self.engine.retrain()
            # Notificar en el hilo principal (after es thread-safe en Tkinter)
            self.after(0, lambda: self._on_retrain_done(name, n_samples))

        threading.Thread(target=_do_retrain, daemon=True).start()

    def _on_retrain_done(self, name: str, n_samples: int):
        messagebox.showinfo(
            "Registro completado",
            f"✓ '{name}' registrado con {n_samples} muestras.\n"
            f"El modelo ya está listo para reconocerlo."
        )
        self.show_view("database")

    # ── Lógica de Video ──

    def _bg_process(self, frame: "np.ndarray"):
        """Corre InsightFace en hilo de fondo para no bloquear la GUI."""
        try:
            result = self.engine.process_frame(frame, camera_name=self.camera_label)
            self._display_frame = result
        except Exception:
            pass
        finally:
            self._processing = False

    def update_video(self):
        """
        Loop principal de video — se llama cada 15 ms via self.after().
        Muestra el último frame procesado inmediatamente (no bloquea),
        y lanza el procesamiento InsightFace en un hilo de fondo.
        """
        frame, frame_id = self.camera.get_frame()

        if frame is not None and frame_id != self.last_frame_id:
            self.last_frame_id = frame_id

            if self.current_view == "monitoring" and not self.is_minimized:
                # Mostrar el último frame procesado sin esperar al nuevo
                if self._display_frame is not None:
                    self.display_frame(self._display_frame, self.video_label)
                else:
                    self.display_frame(frame, self.video_label)

                # Lanzar procesamiento del frame actual si no hay uno en curso
                if not self._processing:
                    self._processing = True
                    threading.Thread(
                        target=self._bg_process,
                        args=(frame.copy(),),
                        daemon=True
                    ).start()

            elif self.current_view == "registration":
                self.handle_auto_registration(frame)
                h, w = frame.shape[:2]
                cv2.rectangle(frame, (w//2-100, h//2-130), (w//2+100, h//2+130), (255, 255, 255), 2)
                self.display_frame(frame, self.video_reg_label)

            elif self.is_minimized:
                if not self._processing:
                    self._processing = True
                    threading.Thread(
                        target=self._bg_process,
                        args=(frame.copy(),),
                        daemon=True
                    ).start()

        self.after(15, self.update_video)

    def handle_liveness_logic(self, frame):
        results = self.engine._last_results
        now = time.time()
        
        if not self.is_verifying:
            # Buscar alguien conocido para iniciar verificación
            for r in results:
                if r['known'] and r['confidence'] > 60:
                    self.is_verifying = True
                    self.verifying_name = r['name']
                    self.verify_start_time = now
                    self.verify_step = 0
                    self.current_challenge = self.challenges[0]
                    break
        else:
            # Verificar progreso
            elapsed = now - self.verify_start_time
            if elapsed > self.verify_timeout:
                self.is_verifying = False
                messagebox.showwarning("Fallo de Seguridad", f"Tiempo de verificación agotado para {self.verifying_name}")
                return

            # Dibujar Overlay de Verificación
            h, w = frame.shape[:2]
            overlay_h = 100
            cv2.rectangle(frame, (0, 0), (w, overlay_h), (30, 30, 30), -1)
            
            # Texto de instrucción
            status_color = (50, 205, 100) # Verde
            msg = f"VERIFICANDO: {self.verifying_name.upper()}"
            instr = f"PASO {self.verify_step + 1}/2: Gire a {self.current_challenge.upper()}"
            timer = f"TIEMPO: {int(self.verify_timeout - elapsed)}s"
            
            cv2.putText(frame, msg, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, instr, (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
            cv2.putText(frame, timer, (w - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 255), 2)

            # Comprobar si el usuario cumple el desafío actual
            match_found = False
            for r in results:
                if r['name'] == self.verifying_name and r['pose'] == self.current_challenge:
                    match_found = True
                    break
            
            if match_found:
                self.verify_step += 1
                if self.verify_step >= len(self.challenges):
                    # Verificación Exitosa
                    self.is_verifying = False
                    # Aquí se podría disparar una acción de apertura de puerta, etc.
                    cv2.rectangle(frame, (0, 0), (w, h), (50, 255, 50), 10)
                    print(f"ACCESO CONCEDIDO: {self.verifying_name}")
                else:
                    self.current_challenge = self.challenges[self.verify_step]
                    self.verify_start_time = now # Reiniciar tiempo para el siguiente paso? o dejar global? 
                    # Lo dejamos global (20s para todo) según pidió el usuario

    def display_frame(self, frame, label_widget):
        w, h = label_widget.winfo_width(), label_widget.winfo_height()
        if w > 10 and h > 10:
            # cv2.resize es ~3× más rápido que PIL LANCZOS para video en vivo
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tk_image  = ImageTk.PhotoImage(image=Image.fromarray(rgb_image))
        label_widget.configure(image=tk_image)
        label_widget.image = tk_image

    # ── System Tray ──

    def setup_tray(self):
        try:
            image = Image.open("assets/logo.png").resize((64, 64))
        except:
            image = Image.new('RGB', (64, 64), color = (73, 109, 137))
            
        menu = (item('Mostrar', self.restore_from_tray), item('Salir', self.quit_app))
        self.tray_icon = pystray.Icon("OmniFace", image, "OmniFace", menu)
        
        # Ejecutar pystray en un hilo para no bloquear tkinter
        threading.Thread(target=self.tray_icon.run, daemon=True).start()

    def minimize_to_tray(self):
        self.withdraw()
        self.is_minimized = True
        
    def restore_from_tray(self, icon=None, item=None):
        self.deiconify()
        self.is_minimized = False

    def quit_app(self, icon=None, item=None):
        self.engine.shutdown()   # liberar pool de hilos
        self.camera.release()
        self.tray_icon.stop()
        self.destroy()
        sys.exit()

if __name__ == "__main__":
    app = OmniFaceApp()
    app.mainloop()
