import streamlit as st
import mediapipe as mp
import numpy as np
from PIL import Image

# MediaPipe の準備
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)  # 信頼度の調整
mp_drawing = mp.solutions.drawing_utils

# StreamlitのWebカメラ入力と写真アップロード
st.title("リアルタイムおよび写真骨格検出")

# カメラ入力
camera_input = st.camera_input("カメラ映像を使用", key="camera")

# 写真アップロード
uploaded_image = st.file_uploader("写真をアップロード", type=["jpg", "jpeg", "png"])

if camera_input is not None:
    # Webカメラの映像を処理
    image = Image.open(camera_input)
    frame = np.array(image)
    st.image(frame, channels="RGB", caption="カメラ映像", use_column_width=True)

    # MediaPipeでポーズ検出
    results = pose.process(frame)

    # 骨格を描画
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                                  connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2))

    # 結果画像をStreamlitに表示
    st.image(frame, channels="RGB", caption="骨格検出結果", use_column_width=True)

elif uploaded_image is not None:
    # アップロードされた画像を処理
    image = Image.open(uploaded_image)
    frame = np.array(image)
    st.image(frame, channels="RGB", caption="アップロードされた画像", use_column_width=True)

    # MediaPipeでポーズ検出
    results = pose.process(frame)

    # 骨格を描画
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                                  connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2))

    # 結果画像をStreamlitに表示
    st.image(frame, channels="RGB", caption="骨格検出結果", use_column_width=True)

else:
    st.warning("カメラ映像または画像をアップロードしてください。")
