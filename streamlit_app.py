import streamlit as st
import mediapipe as mp
import numpy as np
from PIL import Image

# MediaPipeの準備
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# StreamlitのWebカメラ入力
st.title("リアルタイム骨格検出")

# サイドバーでモード選択
mode = st.sidebar.radio("モードを選択", ["リアルタイム", "画像アップロード"])

if mode == "リアルタイム":
    # Webカメラの入力
    camera_input = st.camera_input("カメラ映像を使用", key="camera")

    if camera_input is not None:
        # PIL形式の画像をNumPy配列に変換
        image = Image.open(camera_input)
        frame = np.array(image)

        # MediaPipeでポーズ検出
        results = pose.process(frame)

        # 骨格を描画
        if results.pose_landmarks:
            # ポイントの色を青、サイズを小さく、線を黒に変更
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=5, circle_radius=2),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=2)
            )

        # 結果画像を表示
        st.image(frame, channels="RGB", caption="骨格検出結果", use_container_width=True)

elif mode == "画像アップロード":
    # 画像アップロードの場合
    uploaded_file = st.file_uploader("画像をアップロード", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # 画像を読み込む
        image = Image.open(uploaded_file)
        frame = np.array(image)

        # MediaPipeでポーズ検出
        results = pose.process(frame)

        # 骨格を描画
        if results.pose_landmarks:
            # ポイントの色を青、サイズを小さく、線を黒に変更
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=5, circle_radius=2),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=2)
            )

        # 結果画像を表示
        st.image(frame, channels="RGB", caption="骨格検出結果", use_container_width=True)
