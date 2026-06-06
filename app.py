import streamlit as st
import mediapipe as mp

st.title("MediaPipe Inspection")

st.write("Version:", getattr(mp, "__version__", "Unknown"))
st.write("File:", getattr(mp, "__file__", "Unknown"))

st.write("Available attributes:")

for item in dir(mp):
    st.write(item)
