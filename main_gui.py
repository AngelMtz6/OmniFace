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
        
        # Inicializar componentes
        self.camera_index = 0
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
        self.current_step = 0
        self.captured_samples = []
        self.samples_per_step = 10
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

        self.sidebar_footer = ctk.CTkLabel(self.sidebar, text="Hackatec 2026", font=ctk.CTkFont(size=10))
        self.sidebar_footer.pack(side="bottom", pady=20)

        # ── Main Content Area ──
        self.main_content = ctk.CTkFrame(self, corner_radius=15, fg_color="transparent")
        self.main_content.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)

        # Vistas
        self.init_monitoring_view()
        self.init_database_view()
        self.init_registration_view()
        
        self.show_view("monitoring")

        # ── System Tray ──
        self.setup_tray()

        # Iniciar loop de video
        self.update_video()
        
        # Protocolo de cierre
        self.protocol("WM_DELETE_WINDOW", self.minimize_to_tray)

    # ── Vistas ──

    def init_monitoring_view(self):
        self.view_monitoring = ctk.CTkFrame(self.main_content, fg_color="transparent")
        
        # Feed de Video
        self.video_label = tk.Label(self.view_monitoring, bg="#1a1a1a")
        self.video_label.pack(expand=True, fill="both", padx=10, pady=10)
        
        # Panel de Status
        self.status_panel = ctk.CTkFrame(self.view_monitoring, height=100)
        self.status_panel.pack(fill="x", padx=10, pady=(0, 10))
        
        self.status_text = ctk.CTkLabel(self.status_panel, text="Sistema: Activo", text_color="#50CD64", font=ctk.CTkFont(weight="bold"))
        self.status_text.pack(side="left", padx=20)
        
        self.btn_switch_cam = ctk.CTkButton(self.status_panel, text="Cambiar Cámara", width=120, command=self.switch_camera)
        self.btn_switch_cam.pack(side="right", padx=20)

    def switch_camera(self):
        self.camera_index = (self.camera_index + 1) % 3  # Probar 0, 1, 2
        self.camera.release()
        self.camera._initialized = False  # Resetear para forzar re-init
        self.camera.__init__(video_source=self.camera_index)
        messagebox.showinfo("Cámara", f"Cambiando a fuente de video #{self.camera_index}")

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

        # ── Nombre ──
        name_row = ctk.CTkFrame(self.view_registration, fg_color="transparent")
        name_row.pack(pady=5)
        self.name_entry = ctk.CTkEntry(name_row, placeholder_text="Nombre completo", width=260)
        self.name_entry.pack(side="left", padx=(0, 8))
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
                                              text="Ingresa el nombre y presiona Iniciar",
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

    def show_view(self, view_name):
        self.view_monitoring.pack_forget()
        self.view_database.pack_forget()
        self.view_registration.pack_forget()
        
        if view_name == "monitoring":
            self.view_monitoring.pack(expand=True, fill="both")
        elif view_name == "database":
            self.view_database.pack(expand=True, fill="both")
            self.refresh_identities()
        elif view_name == "registration":
            self.reset_registration_state()
            self.view_registration.pack(expand=True, fill="both")
        
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
            delete_identity(identity_id)
            self.engine.retrain()
            self.refresh_identities()

    # ── Lógica de Registro ──

    def reset_registration_state(self):
        self.current_step = 0
        self.current_step_samples = 0
        self.captured_samples = []
        self.registration_name = ""
        self.is_capturing_auto = False
        if hasattr(self, 'name_entry'):
            self.name_entry.delete(0, 'end')
            self.name_entry.configure(state="normal")
            self.instruction_label.configure(text="Ingresa el nombre y presiona Iniciar", text_color="#AAAAAA")
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
            name = self.name_entry.get().strip()
            if not name:
                messagebox.showwarning("Atención", "Por favor ingresa un nombre")
                return
            self.registration_name = name
            self.name_entry.configure(state="disabled")
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
        """Lógica de captura automática durante el registro."""
        if not self.is_capturing_auto:
            return

        # Pequeña pausa entre poses para que el usuario se mueva
        if hasattr(self, '_last_step_time') and time.time() - self._last_step_time < 3.0:
            self.instruction_label.configure(text=f"¡Prepárate! Siguiente pose...", text_color="#FFA500")
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Intentar detectar rostro en la pose actual
        rects = self.engine.detector.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
        
        # Si no detecta con frontal, intentar con perfil si estamos en esos pasos
        if len(rects) == 0 and self.current_step > 4:
            rects = self.engine.profile_detector.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))

        if len(rects) > 0:
            # ── Cara detectada ──
            self.face_detect_label.configure(text="⬤  ¡Rostro detectado! Capturando…",
                                             text_color="#50CD64")
            self.video_reg_frame.configure(border_color="#50CD64")

            x, y, w, h = rects[0]
            # Alinear + CLAHE + resize 150×150 (mismo pipeline que retrain)
            face = self.engine._prepare_face(gray[y:y+h, x:x+w])
            _, buf = cv2.imencode('.jpg', face)

            self.captured_samples.append(buf.tobytes())
            self.current_step_samples += 1

            if self.current_step_samples >= self.samples_per_step:
                # Paso completado — marcar burbuja verde y pausar
                self._update_step_bubbles(self.current_step)   # marca actual como completado visual
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
            # ── Sin cara ──
            self.face_detect_label.configure(text="⬤  Buscando rostro…", text_color="#FF5555")
            self.video_reg_frame.configure(border_color="#552222")

    def finish_registration(self):
        from app.database import save_identity
        
        save_identity(self.registration_name, self.captured_samples)
        self.engine.retrain()
        
        messagebox.showinfo("Éxito", f"Usuario '{self.registration_name}' registrado con {len(self.captured_samples)} muestras.")
        self.show_view("database")

    # ── Lógica de Video ──

    def update_video(self):
        # Actualizar feed según la vista activa
        frame, frame_id = self.camera.get_frame()
        
        if frame is not None and frame_id != self.last_frame_id:
            self.last_frame_id = frame_id
            
            # Procesar según vista
            if self.current_view == "monitoring" and not self.is_minimized:
                processed = self.engine.process_frame(frame)
                # La verificación Liveness se elimina por ahora según petición
                # self.handle_liveness_logic(processed)
                self.display_frame(processed, self.video_label)
            
            elif self.current_view == "registration":
                # Lógica de captura automática
                self.handle_auto_registration(frame)
                
                # Dibujar un rectángulo guía
                h, w = frame.shape[:2]
                cv2.rectangle(frame, (w//2-100, h//2-130), (w//2+100, h//2+130), (255, 255, 255), 2)
                self.display_frame(frame, self.video_reg_label)
            
            elif self.is_minimized:
                self.engine.process_frame(frame)

        self.after(20, self.update_video)

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
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        
        w, h = label_widget.winfo_width(), label_widget.winfo_height()
        if w > 10 and h > 10:
            pil_image = pil_image.resize((w, h), Image.Resampling.LANCZOS)
        
        tk_image = ImageTk.PhotoImage(image=pil_image)
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
        self.camera.release()
        self.tray_icon.stop()
        self.destroy()
        sys.exit()

if __name__ == "__main__":
    app = OmniFaceApp()
    app.mainloop()
