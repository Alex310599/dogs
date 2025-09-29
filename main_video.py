from ultralytics import YOLO     # Importamos YOLO para la detección
import tempfile                  # Para manejar archivos temporales (guardar el video)
import cv2                       # OpenCV: lectura y procesamiento de frames de video
import streamlit as st           # Streamlit: interfaz web

def main():
    st.title("Detección de Objetos")  # Título de la app
    st.sidebar.title("Configuración") # Barra lateral para configuración
    st.sidebar.markdown("---")
    confidence = st.sidebar.slider("Confidence", min_value=0.0, max_value=1.0, value=0.25)  # Slider para elegir confianza mínima
    st.sidebar.markdown("---")

    video_file_buffer = st.sidebar.file_uploader("Sube un vídeo", type=["mp4", "mov", "avi", "asf", "m4v"])  # Subida de video
    tfflie = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)  # Se guarda en archivo temporal

    vid = ""
    if video_file_buffer is not None:
        tfflie.write(video_file_buffer.read())  # Guardamos el video subido
        vid = cv2.VideoCapture(tfflie.name)     # Lo abrimos con OpenCV
        st.sidebar.text("Reproduciendo Video")
        st.sidebar.video(video_file_buffer)     # Mostramos el video original en la barra lateral

    stframe = st.empty()          # Espacio vacío para ir mostrando frames procesados
    st.sidebar.markdown("---")
    model = YOLO("best.pt")       # Cargamos el modelo YOLO entrenado

    if vid != "":
        while vid.isOpened():     # Mientras el video esté abierto
            success, frame = vid.read()    # Leemos un frame
            if success:
                results = model(frame, conf=confidence)      # Ejecutamos YOLO sobre el frame
                annotated_frame = results[0].plot()          # Dibujamos cajas y etiquetas
                stframe.image(annotated_frame, channels="BGR")  # Mostramos el frame procesado en Streamlit
        vid.release()             # Cerramos el video cuando termina

if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        pass
