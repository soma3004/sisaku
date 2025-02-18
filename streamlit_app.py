import streamlit as st
import mediapipe as mp
import numpy as np
from PIL import Image

# MediaPipe の準備
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)  # 信頼度の調整
mp_drawing = mp.solutions.drawing_utils

# Streamlitのサイドバーでモード選択
st.title("骨格検出アプリ")

mode = st.sidebar.radio("モードを選択", ["カメラモード", "画像アップロードモード"])

if mode == "カメラモード":
    # カメラ入力
    st.subheader("カメラ映像を使用")
    camera_input = st.camera_input("カメラ映像を使用", key="camera")

    if camera_input is not None:
        # Webカメラの映像を処理
        image = Image.open(camera_input)
        frame = np.array(image)
        st.image(frame, channels="RGB", caption="カメラ映像", use_column_width=True)

        # MediaPipeでポーズ検出
        results = pose.process(frame)

        # 骨格を描画
        if results.pose_landmarks:
            # ランドマークの描画（青色、小さな円）
            landmark_spec = mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
            # 接続線の描画（黒色）
            connection_spec = mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=2)
            
            # ランドマークと接続線の描画
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      landmark_drawing_spec=landmark_spec, connection_drawing_spec=connection_spec)

        # 結果画像をStreamlitに表示
        st.image(frame, channels="RGB", caption="骨格検出結果", use_column_width=True)

elif mode == "画像アップロードモード":
    # 画像アップロード
    st.subheader("画像をアップロード")
    uploaded_image = st.file_uploader("写真をアップロード", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # アップロードされた画像を処理
        image = Image.open(uploaded_image)
        frame = np.array(image)
        st.image(frame, channels="RGB", caption="アップロードされた画像", use_column_width=True)

        # MediaPipeでポーズ検出
        results = pose.process(frame)

        # 骨格を描画
        if results.pose_landmarks:
            # ランドマークの描画（青色、小さな円）
            landmark_spec = mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
            # 接続線の描画（黒色）
            connection_spec = mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=2)
            
            # ランドマークと接続線の描画
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      landmark_drawing_spec=landmark_spec, connection_drawing_spec=connection_spec)

        # 結果画像をStreamlitに表示
        st.image(frame, channels="RGB", caption="骨格検出結果", use_column_width=True)
