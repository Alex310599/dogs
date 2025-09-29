# Instalación de librerías necesarias (ejecutar en terminal si no están instaladas)
# pip install streamlit-webrtc
# pip install gdown

# Importamos librerías
import streamlit as st  # Framework para crear apps web interactivas
import cv2  # OpenCV para procesamiento de imágenes y videos
from ultralytics import YOLO  # YOLO para detección de objetos
from streamlit_webrtc import (
    webrtc_streamer,
    VideoTransformerBase,
)  # Para cámara en vivo
import av  # Manejo de frames de video
from PIL import Image  # Para abrir y manipular imágenes
import gdown  # Para descargar archivos desde Google Drive
import os  # Para manejo de archivos y rutas
import asyncio  # Para procesamiento asíncrono
import tempfile  # Para manejo de archivos temporales (videos)


# Función para crear tarjetas visuales en la página principal
def create_card(title, image_url):
    card_html = f"""
    <div class="card">
        <img class="card-image" src="{image_url}" alt="{title}">
        <div class="card-title">{title}</div>
    </div>
    """
    return card_html  # Devuelve el código HTML de la tarjeta


# Función para descargar el modelo YOLO desde Google Drive
def download_model_from_gdrive(gdrive_url, output_path):
    gdown.download(gdrive_url, output_path, quiet=False, fuzzy=True)


# Carga y cachea el modelo para que no se descargue ni recargue cada vez
@st.cache_resource
def load_model():
    model_path = "best.pt"  # Nombre del archivo del modelo
    gdrive_url = "https://drive.google.com/file/d/10J9RGzU8_W70Da78FRRY-Bm8V7RR56O9/view?usp=drive_link"
    if not os.path.exists(model_path):  # Si el modelo no existe, descargarlo
        download_model_from_gdrive(gdrive_url, model_path)
    model = YOLO(model_path)  # Cargar el modelo YOLO
    return model  # Devolver el modelo cargado


# Cargamos el modelo
model = load_model()


# Clase para procesar cada frame de la cámara en tiempo real
class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.model = None
        self.confidence = 0.25  # Confianza mínima por defecto

    # Método para configurar el modelo y la confianza
    def set_params(self, model, confidence):
        self.model = model
        self.confidence = confidence

    # Método que se ejecuta en cada frame recibido de la cámara
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")  # Convertir frame a numpy array
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convertir a RGB para YOLO
        if self.model:  # Si el modelo está cargado
            results = self.model(img_rgb, conf=self.confidence)  # Detectar objetos
            if results:  # Si hay resultados
                annotated_frame = results[0].plot()  # Dibujar cajas y etiquetas
                return av.VideoFrame.from_ndarray(
                    cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR), format="bgr24"
                )
        return av.VideoFrame.from_ndarray(
            img, format="bgr24"
        )  # Si no hay detección, devolver frame original


# Función asíncrona para procesar imágenes subidas
async def process_image(image, model, confidence):
    img = Image.open(image)  # Abrir imagen con PIL
    results = await asyncio.to_thread(
        model, img, conf=confidence
    )  # Ejecutar YOLO de forma asíncrona
    return results  # Devolver resultados de detección


# Función principal de la aplicación
def main():
    st.title("Detección de Objetos")  # Título principal
    activities = [
        "Principal",
        "Usar cámara",
        "Subir imagen",
        "Subir vídeo",
    ]  # Opciones del menú lateral
    choice = st.sidebar.selectbox(
        "Selecciona actividad", activities
    )  # Selector de actividad
    st.sidebar.markdown("---")  # Línea separadora

    # Página principal con tarjetas visuales
    if choice == "Principal":
        st.markdown(
            "<h4 style='color:white;'>Aplicación web de detección de objetos usando YOLO, Streamlit y Python.</h4><br>",
            unsafe_allow_html=True,
        )
        # Crear tres columnas para mostrar tarjetas
        col1, col2, col3 = st.columns(3)
        col1.markdown(
            create_card(
                "Usar cámara",
                "https://st2.depositphotos.com/1915171/5331/v/450/depositphotos_53312473-stock-illustration-webcam-sign-icon-web-video.jpg",
            ),
            unsafe_allow_html=True,
        )
        col2.markdown(
            create_card(
                "Subir imagen",
                "https://i.pinimg.com/736x/e1/91/5c/e1915cea845d5e31e1ec113a34b45fd8.jpg",
            ),
            unsafe_allow_html=True,
        )
        col3.markdown(
            create_card(
                "Subir vídeo",
                "https://static.vecteezy.com/system/resources/previews/005/919/290/original/video-play-film-player-movie-solid-icon-illustration-logo-template-suitable-for-many-purposes-free-vector.jpg",
            ),
            unsafe_allow_html=True,
        )

    # Opción: Usar cámara en tiempo real
    elif choice == "Usar cámara":
        st.header("Utiliza tu cámara")
        if model:  # Si el modelo está cargado
            confidence_slider = st.sidebar.slider(
                "Confidence", min_value=0.0, max_value=1.0, value=0.25
            )
            start_detection = st.checkbox(
                "Iniciar detección de objetos"
            )  # Checkbox para iniciar detección
            video_transformer = VideoTransformer()  # Crear transformer para cámara
            if start_detection:
                video_transformer.set_params(
                    model, confidence_slider
                )  # Configurar modelo y confianza
            # Iniciar transmisión en tiempo real
            webrtc_streamer(
                key="example",
                video_processor_factory=lambda: video_transformer,
                rtc_configuration={
                    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
                },
            )

    # Opción: Subir imagen
    elif choice == "Subir imagen":
        confidence_slider = st.sidebar.slider(
            "Confidence", min_value=0.0, max_value=1.0, value=0.25
        )
        image = st.file_uploader(
            "Sube imagen", type=["png", "jpg", "jpeg", "gif"]
        )  # Subir archivo de imagen
        if image:
            col1, col2, col3 = st.columns([1, 1, 1])  # Crear tres columnas
            col1.image(image, caption="Imagen original")  # Mostrar imagen original
            if model:
                with col2:
                    with st.spinner("Procesando imagen..."):
                        results = asyncio.run(
                            process_image(image, model, confidence_slider)
                        )  # Procesar imagen
                        if results:
                            annotated_frame = results[
                                0
                            ].plot()  # Dibujar cajas y etiquetas
                            annotated_frame = cv2.cvtColor(
                                annotated_frame, cv2.COLOR_RGB2BGR
                            )
                            col2.image(
                                annotated_frame, caption="Imagen anotada"
                            )  # Mostrar imagen procesada
                            for result in results[
                                0
                            ].boxes:  # Recorrer objetos detectados
                                idx = int(
                                    result.cls.cpu().numpy()[0]
                                )  # Clase detectada
                                confidence = result.conf.cpu().numpy()[0]  # Confianza
                        else:
                            col3.write("No se detectaron objetos.")
            else:
                st.error(
                    "El modelo no se cargó correctamente."
                )  # Mensaje si no hay modelo

    # Opción: Subir vídeo
    elif choice == "Subir vídeo":
        st.header("Sube un vídeo para detección de objetos")
        confidence_slider = st.sidebar.slider(
            "Confidence", min_value=0.0, max_value=1.0, value=0.25
        )
        video_file = st.file_uploader(
            "Sube vídeo", type=["mp4", "avi", "mov", "mkv"]
        )  # Subir archivo de vídeo
        if video_file:
            tfile = tempfile.NamedTemporaryFile(delete=False)  # Crear archivo temporal
            tfile.write(video_file.read())  # Guardar contenido del video
            cap = cv2.VideoCapture(tfile.name)  # Abrir video con OpenCV
            col1, col2 = st.columns(2)  # Crear columnas para video procesado y original
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Ancho del video
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Alto del video
            out = cv2.VideoWriter(
                "output.mp4",
                cv2.VideoWriter_fourcc(*"mp4v"),
                30,
                (frame_width, frame_height),
            )  # Guardar video procesado
            stframe = st.empty()  # Espacio para mostrar frames
            progress_bar = st.progress(0)  # Barra de progreso
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Total de frames

            # Columna 1: Mostrar el video procesado frame por frame
            with col1:
                while cap.isOpened():  # Mientras el video tenga frames
                    ret, frame = cap.read()  # Leer un frame
                    if not ret:  # Si no se puede leer más, salir del bucle
                        break
                    img_rgb = cv2.cvtColor(
                        frame, cv2.COLOR_BGR2RGB
                    )  # Convertir BGR a RGB para YOLO
                    if model:  # Si el modelo está cargado
                        results = model(
                            img_rgb, conf=confidence_slider
                        )  # Detectar objetos en el frame
                        if results:  # Si se detectaron objetos
                            annotated_frame = results[
                                0
                            ].plot()  # Dibujar cajas y etiquetas
                            # Guardar el frame anotado en el video de salida
                            out.write(cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))
                            # Mostrar el frame anotado en Streamlit
                            stframe.image(annotated_frame, channels="RGB")

                    # Actualizar la barra de progreso según el número de frames procesados
                    progress_bar.progress(
                        min(cap.get(cv2.CAP_PROP_POS_FRAMES) / frame_count, 1.0)
                    )

                cap.release()  # Liberar el video original
                out.release()  # Liberar el video de salida
                st.success("Procesamiento de vídeo completo.")  # Mensaje de éxito

            # Columna 2: Reproducir el video final procesado
            with col2:
                st.video("output.mp4")  # Mostrar el video resultante


# Ejecutar la aplicación
if __name__ == "__main__":
    main()  # Llamar a la función principal de la app
