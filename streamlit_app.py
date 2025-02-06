import mediapipe as mp
import numpy as np
import streamlit as st
from PIL import Image
def image_face_detection():
    st.write('face detection from image file')
    img_file = st.file_uploader("画像を選択", type="jpg")
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils

    if img_file is not None:
        with st.spinner("検出中"):
            img = Image.open(img_file)
            img = np.array(img, dtype=np.uint8)

            with mp_face_detection.FaceDetection(model_selection=1, 
                                                 min_detection_confidence=0.5) as face_detection:

                results = face_detection.process(img)
                if not results.detections:
                    st.write('Detection failure')
                else:
                    annotated_image = img.copy()
                    for detection in results.detections:
                        mp_drawing.draw_detection(annotated_image, detection)

                    st.image(annotated_image, caption="検出結果")

