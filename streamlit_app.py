import streamlit as st
import mediapipe as mp
import numpy as np
from PIL import Image
import cv2

# MediaPipeの準備
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Pose検出オブジェクトをコンテキストマネージャで初期化
with mp_pose.Pose(
    static_image_mode=False, 
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5
) as pose:

    st.title("リアルタイム骨格検出")

    # サイドバーでモード選択
    mode = st.sidebar.radio("モードを選択", ["リアルタイム", "画像アップロード"])

    def process_frame(frame):
        """
        入力のRGB画像（numpy配列）に対してMediaPipeのポーズ検出を実行し、
        検出結果（骨格）を描画した画像を返す。
        """
        # 必要に応じて画像のチャネルが4(RGBA)の場合はRGBに変換
        if frame.shape[-1] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
        
        # 処理前に書き込み不可にしてパフォーマンス向上
        frame.flags.writeable = False
        results = pose.process(frame)
        frame.flags.writeable = True
        
        # 骨格が検出できた場合、描画する
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=5, circle_radius=2),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=2)
            )
        return frame

    if mode == "リアルタイム":
        # Webカメラからの入力（1枚のスナップショット）
        camera_input = st.camera_input("カメラ映像を使用", key="camera")
        if camera_input is not None:
            image = Image.open(camera_input)
            # PIL画像をRGBのnumpy配列に変換
            frame = np.array(image.convert("RGB"))
            processed_frame = process_frame(frame)
            st.image(processed_frame, channels="RGB", caption="骨格検出結果", use_container_width=True)

    elif mode == "画像アップロード":
        # 画像アップロードの場合
        uploaded_file = st.file_uploader("画像をアップロード", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            frame = np.array(image.convert("RGB"))
            processed_frame = process_frame(frame)
            st.image(processed_frame, channels="RGB", caption="骨格検出結果", use_container_width=True)
