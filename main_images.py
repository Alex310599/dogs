from ultralytics import YOLO
import streamlit as st
import cv2
from PIL import Image

#image_path = " lays3.jpg"

st.title("Ingresa tu imagen para detección")
image = st.file_uploader('Sube imagen', type=["png", "jpg", "jpeg", "gif"])

if image:
    image = Image.open(image)
    st.image(image=image)

    model_path = "best.pt"
    model = YOLO(model_path)
    results = model(image, conf=0.35, stream=False)
    print(results[0].boxes)
    if len(results) > 0:
        result = results[0]
        
        print(result.boxes.cls.cpu().numpy()[0])
        print(result.boxes.conf.cpu().numpy()[0])
        # Save results to disk
        result.save(filename="result.jpg")
        result = Image.open("result.jpg")
        st.image(image=result)
