import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

st.title("Face Mesh Detection")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = np.array(image)

    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        min_detection_confidence=0.5
    ) as face_mesh:

        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if results.multi_face_landmarks:
            custom_spec = mp_drawing.DrawingSpec(
                color=(230, 216, 173),
                thickness=1,
                circle_radius=1
            )

            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=custom_spec,
                    connection_drawing_spec=custom_spec
                )

        st.image(image, caption="Face Mesh Result")