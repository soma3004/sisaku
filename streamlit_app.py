import streamlit as st
import numpy as np
import cv2
import av
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# === Mediapipe セットアップ ===
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose()
hands = mp_hands.Hands()

# === サイドバーでモード選択 ===
mode = st.sidebar.radio("📌 モード選択", ["リアルタイム検出 (PC)", "リアルタイム検出 (スマホ)", "写真から検出"])
sub_mode = st.sidebar.radio("📌 検出対象", ["体の関節のみ", "手の関節のみ", "すべて"])

st.title("📌 PC & スマホ対応 骨格検出アプリ")

# === リアルタイム検出 (スマホ & PC共通処理) ===
def process_frame(img):
    """Mediapipe で骨格・手の関節を検出"""
    results_pose = pose.process(img) if sub_mode in ["体の関節のみ", "すべて"] else None
    results_hands = hands.process(img) if sub_mode in ["手の関節のみ", "すべて"] else None

    if results_pose and results_pose.pose_landmarks:
        mp_drawing.draw_landmarks(img, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    if results_hands and results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:
            mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    return img

# === モード①: PCのWebカメラでリアルタイム検出 ===
if mode == "リアルタイム検出 (PC)":
    st.write("💻 PCのWebカメラを使用します")

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("カメラを開けませんでした。別のアプリが使用していませんか？")
    else:
        stframe = st.empty()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error("カメラから映像を取得できませんでした")
                break

            frame = process_frame(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            stframe.image(frame, channels="RGB")

        cap.release()

# === モード②: スマホのカメラでリアルタイム検出 ===
elif mode == "リアルタイム検出 (スマホ)":
    st.write("📱 スマホのカメラを使用します")

    class VideoProcessor(VideoProcessorBase):
        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            img = process_frame(img)
            return av.VideoFrame.from_ndarray(img, format="bgr24")

    webrtc_streamer(key="example", video_processor_factory=VideoProcessor)

# === モード③: 画像をアップロードして骨格検出 ===
elif mode == "写真から検出":
    st.write("🖼 写真をアップロードして検出します")

    uploaded_file = st.file_uploader("画像を選択", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

        img = process_frame(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        st.image(img, channels="RGB", caption="検出結果")
