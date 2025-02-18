import streamlit as st
import mediapipe as mp
import numpy as np
from PIL import Image

# MediaPipe の準備
mp_pose = mp.solutions.pose
# Pose の信頼度は必要に応じて調整してください
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

st.title("骨格検出アプリ")

# サイドバーでモードの切り替え
mode = st.sidebar.radio("モードを選択してください", ["リアルタイム", "画像アップロード"])

def process_frame(frame):
    """
    入力画像（NumPy配列）に対してMediaPipeでポーズ検出を行い、骨格を描画した画像を返す関数。
    """
    results = pose.process(frame)
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
        )
    return frame

if mode == "リアルタイム":
    # Webカメラ入力
    camera_input = st.camera_input("カメラ映像を使用", key="camera")
    if camera_input is not None:
        # PIL形式の画像をRGBに変換し、NumPy配列へ
        image = Image.open(camera_input).convert("RGB")
        frame = np.array(image)
        
        # 処理用に画像のコピーを作成して、骨格検出を実施
        processed_frame = process_frame(frame.copy())
        
        # 結果を表示
        st.image(processed_frame, channels="RGB", use_column_width=True, caption="骨格検出結果")

elif mode == "画像アップロード":
    # 画像アップロード
    uploaded_file = st.file_uploader("画像をアップロード", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        frame = np.array(image)
        
        processed_frame = process_frame(frame.copy())
        st.image(processed_frame, channels="RGB", use_column_width=True, caption="骨格検出結果")
