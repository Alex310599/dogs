# pip install streamlit-webrtc
# pip install gdown

import streamlit as st
import cv2
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
import gdown
import os


# Function to download the model from Google Drive
def download_model_from_gdrive(gdrive_url, output_path):
    gdown.download(gdrive_url, output_path, quiet=False, fuzzy=True)


# Cache the model loading
@st.cache_resource
def load_model():
    model_path = "best.pt"
    gdrive_url = "https://drive.google.com/file/d/10J9RGzU8_W70Da78FRRY-Bm8V7RR56O9/view?usp=drive_link"
    if not os.path.exists(model_path):
        download_model_from_gdrive(gdrive_url, model_path)
    model = YOLO(model_path)
    return model


model = load_model()


class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.model = None
        self.confidence = 0.25

    def set_params(self, model, confidence):
        self.model = model
        self.confidence = confidence

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.model:
            results = self.model(img_rgb, conf=self.confidence)
            if results:
                annotated_frame = results[0].plot()
                return av.VideoFrame.from_ndarray(
                    cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR), format="bgr24"
                )
        return av.VideoFrame.from_ndarray(img, format="bgr24")


def main():
    st.title("Detección de Objetos")
    activities = [
        "Usar cámara",
    ]
    choice = st.sidebar.selectbox("Selecciona actividad", activities)
    st.sidebar.markdown("---")

    if choice == "Usar cámara":
        st.header("Utiliza tu cámara")
        if model:
            confidence_slider = st.sidebar.slider(
                "Confidence", min_value=0.0, max_value=1.0, value=0.25
            )
            start_detection = st.checkbox("Iniciar detección de objetos")
            video_transformer = VideoTransformer()
            if start_detection:
                video_transformer.set_params(model, confidence_slider)
            webrtc_streamer(
                key="example",
                video_processor_factory=lambda: video_transformer,
                rtc_configuration={
                    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
                },
            )


if __name__ == "__main__":
    main()
