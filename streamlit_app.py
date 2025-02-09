import streamlit as st
import mediapipe as mp
import numpy as np
import cv2
from PIL import Image
import pandas as pd

# Mediapipeのポーズ推定を初期化
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils  # 骨格描画用

st.title("Pose Detection with Streamlit")

# カメラ画像を取得
img_file = st.camera_input("Take a picture")

if img_file is not None:
    # 画像をPILからOpenCVの形式に変換
    image = Image.open(img_file)
    frame = np.array(image)

    # OpenCVはBGR、StreamlitはRGBを扱うため変換
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Mediapipeでポーズ推定
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # 座標データを格納するリスト
    landmarks_list = []

    # 骨格点を描画
    if results.pose_landmarks:
        for idx, landmark in enumerate(results.pose_landmarks.landmark):
            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            landmarks_list.append({"Point": idx, "X": x, "Y": y})

        # 骨格ラインの描画
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Streamlitで画像を表示
    st.image(frame, channels="BGR", use_column_width=True)

    # 座標データを表で表示
    if landmarks_list:
        df = pd.DataFrame(landmarks_list)
        st.table(df)
