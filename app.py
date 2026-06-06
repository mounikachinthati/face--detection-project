import streamlit as st
import mediapipe as mp
import sys

st.write("Python:", sys.version)
st.write("MediaPipe version:", getattr(mp, "__version__", "Unknown"))
st.write("MediaPipe file:", getattr(mp, "__file__", "Unknown"))
st.write("Has solutions:", hasattr(mp, "solutions"))
st.write("Attributes:", dir(mp))
