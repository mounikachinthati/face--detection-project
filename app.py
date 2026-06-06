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

    image = Image.open(uploaded_file).convert("RGB")
    image = np.array(image)

    st.image(
        image,
        caption="Uploaded Image",
        width="stretch"
    )

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = cv2.equalizeHist(gray)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    st.write("Cascade loaded:", not face_cascade.empty())

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=3,
        minSize=(20, 20)
    )

    st.subheader("Detection Results")
    st.write("Faces detected:", len(faces))
    st.write("Image shape:", image.shape)

    result = image.copy()

    for (x, y, w, h) in faces:
        cv2.rectangle(
            result,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            3
        )

    st.image(
        result,
        caption="Detected Faces",
        width="stretch"
    )
