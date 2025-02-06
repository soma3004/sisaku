import streamlit as st
import mediapipe as mp
import numpy as np
from PIL import Image
import cv2

# MediaPipe の準備
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# StreamlitのWebカメラ入力
st.title("リアルタイム骨格検出")

# Streamlitのカメラ入力
camera_input = st.camera_input("カメラ映像を使用", key="camera")

if camera_input is not None:
    # カメラ映像のフレームを取得
    frame = np.array(camera_input)

    # MediaPipeでポーズ検出
    results = pose.process(frame)

    # 骨格を描画
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # 結果画像をStreamlitに表示
    st.image(frame, channels="RGB", use_column_width=True)
