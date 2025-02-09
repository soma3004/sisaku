import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import threading
from PIL import Image

# Mediapipeのセットアップ
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()
hands = mp_hands.Hands()

# サイドバーでモード選択
main_mode = st.sidebar.radio("🔍 検出モードを選択", ["リアルタイム検出", "写真から座標を検出", "動画から座標を検出"], key="main_mode_key")
sub_mode = st.sidebar.radio("📌 検出対象を選択", ["体の関節のみ", "手の関節のみ", "すべて"], key="sub_mode_key")

st.title("📌 骨格検出アプリ")

# **リアルタイム検出モード**
if main_mode == "リアルタイム検出":
    st.subheader("🎥 リアルタイムで骨格を検出中...")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # バッファを最小化
    FRAME_WINDOW = st.empty()
    
    def process_video():
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results_pose = pose.process(frame) if sub_mode in ["体の関節のみ", "すべて"] else None
            results_hands = hands.process(frame) if sub_mode in ["手の関節のみ", "すべて"] else None
            if results_pose and results_pose.pose_landmarks:
                mp_drawing.draw_landmarks(frame, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            if results_hands and results_hands.multi_hand_landmarks:
                for hand_landmarks in results_hands.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            FRAME_WINDOW.image(frame, channels="RGB")
        cap.release()
    
    thread = threading.Thread(target=process_video)
    thread.start()

# **写真から座標を検出するモード**
elif main_mode == "写真から座標を検出":
    st.subheader("📸 写真を撮影 or アップロードしてください")
    img_file = st.file_uploader("画像をアップロード", type=["jpg", "jpeg", "png"])
    if img_file:
        image = Image.open(img_file)
        frame = np.array(image)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        results_pose = pose.process(frame) if sub_mode in ["体の関節のみ", "すべて"] else None
        results_hands = hands.process(frame) if sub_mode in ["手の関節のみ", "すべて"] else None
        if results_pose and results_pose.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        if results_hands and results_hands.multi_hand_landmarks:
            for hand_landmarks in results_hands.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        st.image(frame, channels="BGR", use_column_width=True)

# **動画から座標を検出するモード**
elif main_mode == "動画から座標を検出":
    st.subheader("🎞️ 動画をアップロードしてください")
    video_file = st.file_uploader("動画をアップロード", type=["mp4", "mov", "avi"])
    if video_file:
        temp_video_path = "temp_video.mp4"
        with open(temp_video_path, "wb") as f:
            f.write(video_file.read())
        cap = cv2.VideoCapture(temp_video_path)
        FRAME_WINDOW = st.empty()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results_pose = pose.process(frame) if sub_mode in ["体の関節のみ", "すべて"] else None
            results_hands = hands.process(frame) if sub_mode in ["手の関節のみ", "すべて"] else None
            if results_pose and results_pose.pose_landmarks:
                mp_drawing.draw_landmarks(frame, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            if results_hands and results_hands.multi_hand_landmarks:
                for hand_landmarks in results_hands.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            FRAME_WINDOW.image(frame, channels="RGB")
        cap.release()
