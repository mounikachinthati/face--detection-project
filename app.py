import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

st.title("Face Mesh Detection")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        min_detection_confidence=0.5
    ) as face_mesh:

        results = face_mesh.process(image_np)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=image_np,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing.DrawingSpec(
                        thickness=1,
                        circle_radius=1
                    )
                )

            st.image(
                image_np,
                caption="Face Mesh Result",
                use_container_width=True
            )
        else:
            st.warning("No face detected in the image.")
