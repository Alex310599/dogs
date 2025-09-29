# Importamos las librerías necesarias
from ultralytics import YOLO   # Modelo YOLO para detección de objetos
import streamlit as st         # Framework para crear apps web interactivas
import cv2                     # OpenCV (procesamiento de imágenes, aquí opcional)
from PIL import Image          # PIL para abrir y manipular imágenes

# Título de la aplicación
st.title("Ingresa tu imagen para detección")

# Opción para subir una imagen desde el navegador
image = st.file_uploader('Sube imagen', type=["png", "jpg", "jpeg", "gif"])

# Si el usuario subió una imagen...
if image:
    image = Image.open(image)   # Abrir la imagen con PIL
    st.image(image=image)       # Mostrar la imagen original en la app

    # Cargar el modelo YOLO previamente entrenado (best.pt debe estar en la carpeta)
    model_path = "best.pt"
    model = YOLO(model_path)

    # Ejecutar detección sobre la imagen con confianza mínima del 35%
    results = model(image, conf=0.35, stream=False)

    # Mostrar en consola todas las cajas detectadas
    print(results[0].boxes)

    # Si hubo detecciones...
    if len(results) > 0:
        result = results[0]  # Tomamos el primer resultado (la imagen procesada)
        
        # Imprimir en consola la clase y la confianza del primer objeto detectado
        print(result.boxes.cls.cpu().numpy()[0])   # Clase detectada (ej: 0=persona)
        print(result.boxes.conf.cpu().numpy()[0])  # Confianza (ej: 0.87)

        # Guardar la imagen con las detecciones (cajas + etiquetas)
        result.save(filename="result.jpg")

        # Abrir esa imagen ya procesada y mostrarla en la app
        result = Image.open("result.jpg")
        st.image(image=result)
