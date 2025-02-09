import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from PIL import Image

# Mediapipeのセットアップ
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose()
hands = mp_hands.Hands()

# **サイドバーでモード選択**
main_mode = st.sidebar.radio("🔍 検出モードを選択", ["リアルタイム検出", "写真から座標を検出"])
sub_mode = st.sidebar.radio("📌 検出対象を選択", ["体の関節のみ", "手の関節のみ", "すべて"])

st.title("📌 骨格検出アプリ")

# **リアルタイム検出モード**
if main_mode == "リアルタイム検出":
    st.subheader("🎥 リアルタイムで骨格を検出中...")
    cap = cv2.VideoCapture(0)
    FRAME_WINDOW = st.image([])

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("カメラを開けませんでした。")
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # **Mediapipeで処理**
        results_pose = pose.process(frame) if sub_mode in ["体の関節のみ", "すべて"] else None
        results_hands = hands.process(frame) if sub_mode in ["手の関節のみ", "すべて"] else None

        # **体の関節を描画**
        if results_pose and results_pose.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # **手の関節を描画**
        if results_hands and results_hands.multi_hand_landmarks:
            for hand_landmarks in results_hands.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # **映像を更新**
        FRAME_WINDOW.image(frame)

    cap.release()
    cv2.destroyAllWindows()

# **写真から座標を検出するモード**
else:
    st.subheader("📸 写真を撮影 or アップロードしてください")
    img_file = st.file_uploader("画像をアップロード", type=["jpg", "jpeg", "png"])

    if img_file is not None:
        image = Image.open(img_file)
        frame = np.array(image)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # **Mediapipeで処理**
        results_pose = pose.process(frame) if sub_mode in ["体の関節のみ", "すべて"] else None
        results_hands = hands.process(frame) if sub_mode in ["手の関節のみ", "すべて"] else None

        landmarks_list = []

        # **体の関節を描画**
        if results_pose and results_pose.pose_landmarks:
            for idx, landmark in enumerate(results_pose.pose_landmarks.landmark):
                x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                landmarks_list.append({"Type": "Body", "Point": f"Joint_{idx}", "X": x, "Y": y})
            mp_drawing.draw_landmarks(frame, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # **手の関節を描画**
        if results_hands and results_hands.multi_hand_landmarks:
            for hand_landmarks in results_hands.multi_hand_landmarks:
                for idx, landmark in enumerate(hand_landmarks.landmark):
                    x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                    landmarks_list.append({"Type": "Hand", "Point": f"Hand_{idx}", "X": x, "Y": y})
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # **画像を表示**
        st.image(frame, channels="BGR", use_column_width=True)

        # **座標データを表で表示**
        if landmarks_list:
            df = pd.DataFrame(landmarks_list)
            st.table(df)
