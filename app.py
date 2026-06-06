import streamlit as st
import sys

st.title("Debug Info")

st.write("Python:", sys.version)

try:
    import cv2
    st.success("OpenCV imported successfully")
    st.write("OpenCV version:", cv2.__version__)
except Exception as e:
    st.error(f"OpenCV Error: {e}")

try:
    import mediapipe as mp
    st.success("MediaPipe imported successfully")
    st.write("MediaPipe version:", mp.__version__)
except Exception as e:
    st.error(f"MediaPipe Error: {e}")
