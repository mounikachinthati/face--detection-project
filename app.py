import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("Face Mesh Detection")

try:
    import mediapipe as mp

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"]
    )

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

            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_image)

            if results.multi_face_landmarks:
                drawing_spec = mp_drawing.DrawingSpec(
                    thickness=1,
                    circle_radius=1
                )

                for face_landmarks in results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=drawing_spec,
                        connection_drawing_spec=drawing_spec
                    )

                st.image(image, caption="Face Mesh Result")
            else:
                st.warning("No face detected in the image.")

except Exception as e:
    st.error(f"MediaPipe Error: {e}")
