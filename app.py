import streamlit as st
import numpy as np
from PIL import Image
import cv2

st.set_page_config(page_title="Face Detection", page_icon="📷")

st.title("Face Detection App")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = np.array(image)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=3,
        minSize=(20, 20)
    )

    st.write("Faces detected:", len(faces))

    for (x, y, w, h) in faces:
        cv2.rectangle(
            image,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    st.image(
        image_rgb,
        caption="Detected Faces",
        use_container_width=True
    )
