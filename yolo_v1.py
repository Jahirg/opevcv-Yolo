
import sys
import cv2

# importar ANTES de PyQt5: en Windows, si PyQt5 carga
# primero, a veces choca con las DLLs de torch (c10.dll)
# Mantener este orden de carga de las bibliotecas.
from ultralytics import YOLO  

########################################################################
# Si presenta confictos con los .DLL 
# Descarga e instala Microsoft Visual C++ Redistributable
# https://aka.ms/vc14/vc_redist.x64.exe
########################################################################
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap

########################################################################
# TESTED ON
########################################################################
#python 						3.12.10
#opencv-python					5.0.0.3
#numpy							2.5.2
#matplotlib						3.9.4
#PyQt5							5.15.11
#untralytics					8.4.142
########################################################################

MODEL_PATH = "yolo26n.pt"  
#https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt

# MODEL_PATH ="yolov8n.pt"
#https://huggingface.co/Ultralytics/YOLOv8/resolve/main/yolov8n.pt

class YoloApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Deteccion YOLO - Foto / Video")
        self.resize(1100, 650)

        # Estado
        self.model = None
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_frame)

        self._build_ui()
        self._load_model()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # ---------- Panel izquierdo: controles ----------
        controls = QWidget()
        controls.setFixedWidth(220)
        controls_layout = QVBoxLayout(controls)
        controls_layout.setAlignment(Qt.AlignTop)

        # Selector de modo
        mode_box = QGroupBox("Modo")
        mode_layout = QVBoxLayout(mode_box)
        self.radio_foto = QRadioButton("Foto")
        self.radio_video = QRadioButton("Video (webcam)")
        self.radio_foto.setChecked(True)
        self.mode_group = QButtonGroup(self)
        self.mode_group.addButton(self.radio_foto)
        self.mode_group.addButton(self.radio_video)
        mode_layout.addWidget(self.radio_foto)
        mode_layout.addWidget(self.radio_video)
        controls_layout.addWidget(mode_box)

        self.radio_foto.toggled.connect(self._on_mode_changed)

        # Botones foto
        self.btn_load_image = QPushButton("Cargar imagen")
        self.btn_load_image.clicked.connect(self._load_image)
        controls_layout.addWidget(self.btn_load_image)

        # Botones video
        self.btn_start_cam = QPushButton("Iniciar webcam")
        self.btn_start_cam.clicked.connect(self._start_webcam)
        controls_layout.addWidget(self.btn_start_cam)

        self.btn_stop_cam = QPushButton("Detener webcam")
        self.btn_stop_cam.clicked.connect(self._stop_webcam)
        controls_layout.addWidget(self.btn_stop_cam)

        controls_layout.addStretch()

        self.status_label = QLabel("Cargando modelo...")
        self.status_label.setWordWrap(True)
        controls_layout.addWidget(self.status_label)

        main_layout.addWidget(controls)

        # ---------- Panel derecho: visualizacion ----------
        self.display_label = QLabel("Sin imagen")
        self.display_label.setAlignment(Qt.AlignCenter)
        self.display_label.setStyleSheet("background-color: #202020; color: #aaaaaa;")
        self.display_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.display_label.setMinimumSize(400, 400)
        main_layout.addWidget(self.display_label, stretch=1)

        self._on_mode_changed()

    def _on_mode_changed(self):
        foto_mode = self.radio_foto.isChecked()
        self.btn_load_image.setEnabled(foto_mode)
        self.btn_start_cam.setEnabled(not foto_mode)
        self.btn_stop_cam.setEnabled(not foto_mode)
        # Si estaba corriendo la webcam y el usuario cambia a modo foto, se detiene sola
        if foto_mode:
            self._stop_webcam()

    # ------------------------------------------------------------------
    # Modelo
    # ------------------------------------------------------------------
    def _load_model(self):
        try:
            self.model = YOLO(MODEL_PATH)
            self.status_label.setText("Modelo cargado. Listo.")
        except Exception as exc:
            self.status_label.setText("Error cargando el modelo.")
            QMessageBox.critical(self, "Error", f"No se pudo cargar el modelo YOLO:\n{exc}")

    # ------------------------------------------------------------------
    # Modo Foto
    # ------------------------------------------------------------------
    def _load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Seleccionar imagen",
            "",
            "Imagenes (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp)"
        )
        if not file_path:
            return

        photo = cv2.imread(file_path)
        if photo is None:
            QMessageBox.warning(self, "Error", "No se pudo leer la imagen seleccionada.")
            return

        self.status_label.setText("Procesando imagen...")
        QApplication.processEvents()

        result = self.model(photo)
        annotated = result[0].plot()  # BGR con las detecciones dibujadas

        self._show_frame(annotated)
        self.status_label.setText(f"Imagen cargada: {file_path.split('/')[-1]}")

    # ------------------------------------------------------------------
    # Modo Video (webcam)
    # ------------------------------------------------------------------
    def _start_webcam(self):
        if self.cap is not None:
            return  # ya esta corriendo

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "Error", "No se pudo abrir la webcam.")
            self.cap = None
            return

        self.status_label.setText("Webcam activa.")
        self.timer.start(30)  # ~33 fps de refresco de UI

    def _stop_webcam(self):
        if self.timer.isActive():
            self.timer.stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.status_label.setText("Webcam detenida.")

    def _update_frame(self):
        if self.cap is None:
            return
        ok, frame = self.cap.read()
        if not ok:
            self.status_label.setText("No se pudo leer frame de la webcam.")
            return

        result = self.model(frame, verbose=False)
        annotated = result[0].plot()
        self._show_frame(annotated)

    # ------------------------------------------------------------------
    # Utilidad: mostrar frame BGR en el QLabel
    # ------------------------------------------------------------------
    def _show_frame(self, frame_bgr: np.ndarray):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        qimg = QImage(frame_rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        pixmap = pixmap.scaled(
            self.display_label.width(),
            self.display_label.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.display_label.setPixmap(pixmap)

    # ------------------------------------------------------------------
    # Cierre limpio
    # ------------------------------------------------------------------
    def closeEvent(self, event):
        self._stop_webcam()
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = YoloApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
