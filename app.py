import streamlit as st
import cv2
import numpy as np
from PIL import Image
import urllib.request
import os

st.set_page_config(page_title="Face Detection", page_icon="📷")

st.title("Face Detection App")

# Download DNN model files if missing
MODEL_FILE = "res10_300x300_ssd_iter_140000.caffemodel"
CONFIG_FILE = "deploy.prototxt"

if not os.path.exists(MODEL_FILE):
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
        MODEL_FILE
    )

if not os.path.exists(CONFIG_FILE):
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
        CONFIG_FILE
    )

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

    h, w = image.shape[:2]

    net = cv2.dnn.readNetFromCaffe(
        CONFIG_FILE,
        MODEL_FILE
    )

    blob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)),
        1.0,
        (300, 300),
        (104.0, 177.0, 123.0)
    )

    net.setInput(blob)
    detections = net.forward()

    result = image.copy()
    face_count = 0

    for i in range(detections.shape[2]):

        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:

            box = detections[0, 0, i, 3:7] * np.array(
                [w, h, w, h]
            )

            (x1, y1, x2, y2) = box.astype("int")

            cv2.rectangle(
                result,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            face_count += 1

    st.write("Faces detected:", face_count)

    st.image(
        result,
        caption="Detected Faces",
        width="stretch"
    )
