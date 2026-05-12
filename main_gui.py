import tkinter as tk
from tkinter import messagebox
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
from app.database import get_all_identities, delete_identity, get_access_logs, get_stats

# Configuración estética
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class OmniFaceApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("OmniFace - Sistema de Reconocimiento Facial")
        self.geometry("1100x700")
        
        # Inicializar componentes
        self.camera = VideoCamera(video_source=1)
        self.engine = RecognitionEngine()
        
        # Variables de estado
        self.last_frame_id = -1
        self.current_view = "monitoring"
        self.is_minimized = False
        
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

    def init_database_view(self):
        self.view_database = ctk.CTkFrame(self.main_content, fg_color="transparent")
        
        self.db_title = ctk.CTkLabel(self.view_database, text="Gestión de Identidades", font=ctk.CTkFont(size=24, weight="bold"))
        self.db_title.pack(pady=20)
        
        self.identities_list = ctk.CTkScrollableFrame(self.view_database, width=600, height=400)
        self.identities_list.pack(pady=10, padx=20, expand=True, fill="both")
        
        self.btn_refresh = ctk.CTkButton(self.view_database, text="Actualizar Lista", command=self.refresh_identities)
        self.btn_refresh.pack(pady=10)

    def show_view(self, view_name):
        self.view_monitoring.pack_forget()
        self.view_database.pack_forget()
        
        if view_name == "monitoring":
            self.view_monitoring.pack(expand=True, fill="both")
        elif view_name == "database":
            self.view_database.pack(expand=True, fill="both")
            self.refresh_identities()
        
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

    # ── Lógica de Video ──

    def update_video(self):
        if self.current_view == "monitoring" and not self.is_minimized:
            frame, frame_id = self.camera.get_frame()
            
            if frame is not None and frame_id != self.last_frame_id:
                self.last_frame_id = frame_id
                
                # Procesar reconocimiento
                processed = self.engine.process_frame(frame)
                
                # Convertir para Tkinter
                rgb_image = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_image)
                
                # Redimensionar para ajustar al contenedor manteniendo aspecto
                w, h = self.video_label.winfo_width(), self.video_label.winfo_height()
                if w > 10 and h > 10:
                    pil_image = pil_image.resize((w, h), Image.Resampling.LANCZOS)
                
                tk_image = ImageTk.PhotoImage(image=pil_image)
                
                self.video_label.configure(image=tk_image)
                self.video_label.image = tk_image
        
        # Ejecutar reconocimiento en background si está minimizado (opcional)
        elif self.is_minimized:
            frame, frame_id = self.camera.get_frame()
            if frame is not None and frame_id != self.last_frame_id:
                self.last_frame_id = frame_id
                self.engine.process_frame(frame)

        self.after(20, self.update_video)

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
