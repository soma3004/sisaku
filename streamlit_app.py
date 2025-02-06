import streamlit as st
import mediapipe as mp
import numpy as np
from PIL import Image

# MediaPipe の準備
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# StreamlitのWebカメラ入力
st.title("リアルタイム骨格検出")

# Streamlitのカメラ入力
camera_input = st.camera_input("カメラ映像を使用", key="camera")

if camera_input is not None:
    # 画像をPILからNumPy配列に変換
    image = Image.open(camera_input)
    frame = np.array(image)

    # MediaPipeでポーズ検出
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = pose.process(rgb_frame)

    # 骨格を描画
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # 結果画像をStreamlitに表示
    st.image(frame, channels="RGB", use_column_width=True)
