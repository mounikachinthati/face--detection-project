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

    # Load image
    image = Image.open(uploaded_file).convert("RGB")
    image = np.array(image)

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Improve contrast
    gray = cv2.equalizeHist(gray)

    # Load face detector
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=8,
        minSize=(120, 120)
    )

    st.subheader("Detection Results")
    st.write("Faces detected:", len(faces))

    # Draw rectangles
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
        use_container_width=True
    )
