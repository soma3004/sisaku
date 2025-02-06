import cv2
import mediapipe as mp
import streamlit as st
import numpy as np

# MediaPipe の準備
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# StreamlitのWebカメラ入力
st.title("リアルタイム骨格検出")

# Streamlit での画像表示
image_placeholder = st.empty()

# Webカメラからの映像キャプチャ
cap = cv2.VideoCapture(0)

while True:
    # カメラからフレームを読み取る
    ret, frame = cap.read()
    if not ret:
        break

    # BGR画像からRGB画像に変換
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # MediaPipeでポーズ検出
    results = pose.process(rgb_frame)

    # 骨格を描画
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # OpenCVの画像をStreamlitに表示
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_placeholder.image(frame, channels="RGB", use_column_width=True)

    # 'q'キーを押すと終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# カメラのキャプチャを解放
cap.release()
cv2.destroyAllWindows()
