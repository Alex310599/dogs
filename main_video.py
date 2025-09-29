from ultralytics import YOLO
import tempfile
import cv2
import streamlit as st


def main():
    st.title("Detección de Objetos")
    st.sidebar.title("Configuración")
    st.sidebar.markdown("---")
    confidence = st.sidebar.slider(
        "Confidence", min_value=0.0, max_value=1.0, value=0.25
    )
    st.sidebar.markdown("---")

    video_file_buffer = st.sidebar.file_uploader(
        "Sube un vídeo", type=["mp4", "mov", "avi", "asf", "m4v"]
    )
    tfflie = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)

    vid = ""
    if video_file_buffer is not None:
        tfflie.write(video_file_buffer.read())
        vid = cv2.VideoCapture(tfflie.name)

        st.sidebar.text("Reproduciendo Video")
        st.sidebar.video(video_file_buffer)

    stframe = st.empty()
    st.sidebar.markdown("---")
    model = YOLO("best.pt")

    if vid != "":
        while vid.isOpened():
            success, frame = vid.read()
            if success:
                # Run YOLOv8 inference on the frame
                results = model(frame, conf=confidence)
                # Visualize the results on the framd
                annotated_frame = results[0].plot()
                stframe.image(annotated_frame, channels="BGR")

        vid.release()


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        pass
