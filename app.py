import streamlit as st
import mediapipe as mp

st.title("MediaPipe Debug")

st.write("MediaPipe module:", mp)
st.write("MediaPipe file:", getattr(mp, "__file__", "Not found"))
st.write("MediaPipe version:", getattr(mp, "__version__", "No version"))

try:
    st.write("Has solutions:", hasattr(mp, "solutions"))
    
    if hasattr(mp, "solutions"):
        st.success("MediaPipe solutions found!")
        st.write("Face Mesh module:", mp.solutions.face_mesh)
    else:
        st.error("MediaPipe solutions NOT found!")

except Exception as e:
    st.error(f"Error: {e}")
