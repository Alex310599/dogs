import streamlit as st  # Framework web interactivo
import cv2  # OpenCV para procesar imágenes/video
from ultralytics import YOLO  # YOLO para detección de objetos
from streamlit_webrtc import (
    webrtc_streamer,
    VideoTransformerBase,
)  # WebRTC para cámara en vivo
import av  # Para manejar frames de video
import gdown  # Para descargar archivos desde Google Drive
import os  # Manejo de archivos y rutas


# Función para descargar el modelo desde Google Drive
def download_model_from_gdrive(gdrive_url, output_path):
    gdown.download(gdrive_url, output_path, quiet=False, fuzzy=True)


# Cachea la carga del modelo para no descargarlo o cargarlo cada vez
@st.cache_resource
def load_model():
    model_path = "best.pt"  # Nombre del archivo del modelo
    gdrive_url = "https://drive.google.com/file/d/10J9RGzU8_W70Da78FRRY-Bm8V7RR56O9/view?usp=drive_link"
    # Si no existe el modelo, descargarlo
    if not os.path.exists(model_path):
        download_model_from_gdrive(gdrive_url, model_path)
    model = YOLO(model_path)  # Cargar el modelo YOLO
    return model


# Cargamos el modelo
model = load_model()


# Clase para procesar cada frame de la cámara
class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.model = None
        self.confidence = 0.25  # Confianza mínima por defecto

    # Configura el modelo y la confianza
    def set_params(self, model, confidence):
        self.model = model
        self.confidence = confidence

    # Se ejecuta en cada frame recibido de la cámara
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")  # Convertir frame a numpy array
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convertir a RGB para YOLO
        if self.model:
            results = self.model(img_rgb, conf=self.confidence)  # Detectar objetos
            if results:
                annotated_frame = results[0].plot()  # Dibujar cajas en la imagen
                return av.VideoFrame.from_ndarray(
                    cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR), format="bgr24"
                )
        # Si no hay detección, devolver el frame original
        return av.VideoFrame.from_ndarray(img, format="bgr24")


# Función principal de la app
def main():
    st.title("Detección de Objetos")  # Título de la aplicación
    activities = ["Usar cámara"]  # Lista de actividades disponibles
    choice = st.sidebar.selectbox("Selecciona actividad", activities)
    st.sidebar.markdown("---")

    if choice == "Usar cámara":
        st.header("Utiliza tu cámara")  # Encabezado principal
        if model:
            # Slider para ajustar la confianza mínima
            confidence_slider = st.sidebar.slider(
                "Confidence", min_value=0.0, max_value=1.0, value=0.25
            )
            # Checkbox para iniciar la detección
            start_detection = st.checkbox("Iniciar detección de objetos")
            video_transformer = VideoTransformer()
            if start_detection:
                video_transformer.set_params(
                    model, confidence_slider
                )  # Configurar el transformer
            # Iniciar la transmisión de video en tiempo real
            webrtc_streamer(
                key="example",
                video_processor_factory=lambda: video_transformer,
                rtc_configuration={
                    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
                },
            )


# Ejecutar la app
if __name__ == "__main__":
    main()
