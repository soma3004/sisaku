import streamlit as st
import mediapipe as mp
import numpy as np
import cv2
from PIL import Image
import pandas as pd

# Mediapipeのモジュールを初期化
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils  # 骨格描画用
pose = mp_pose.Pose(model_complexity=1, enable_segmentation=False)  # 顔の検出はしない
hands = mp_hands.Hands()

# **顔を除外した体の関節リスト**
BODY_LANDMARKS = [
    mp_pose.PoseLandmark.LEFT_SHOULDER,
    mp_pose.PoseLandmark.RIGHT_SHOULDER,
    mp_pose.PoseLandmark.LEFT_ELBOW,
    mp_pose.PoseLandmark.RIGHT_ELBOW,
    mp_pose.PoseLandmark.LEFT_WRIST,
    mp_pose.PoseLandmark.RIGHT_WRIST,
    mp_pose.PoseLandmark.LEFT_HIP,
    mp_pose.PoseLandmark.RIGHT_HIP,
    mp_pose.PoseLandmark.LEFT_KNEE,
    mp_pose.PoseLandmark.RIGHT_KNEE,
    mp_pose.PoseLandmark.LEFT_ANKLE,
    mp_pose.PoseLandmark.RIGHT_ANKLE
]

st.title("Pose and Hand Tracking (No Face)")

# カメラ画像を取得
img_file = st.camera_input("Take a picture")

if img_file is not None:
    # 画像をPILからOpenCVの形式に変換
    image = Image.open(img_file)
    frame = np.array(image)

    # OpenCVはBGR、StreamlitはRGBを扱うため変換
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Mediapipeでポーズ推定（顔のランドマークなし）
    pose_results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    hand_results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # 座標データを格納するリスト
    landmarks_list = []

    # **体の関節のみを描画（顔は無視）**
    if pose_results.pose_landmarks:
        for landmark_id in BODY_LANDMARKS:
            landmark = pose_results.pose_landmarks.landmark[landmark_id]
            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)  # 緑色で描画
            landmarks_list.append({"Point": landmark_id.name, "X": x, "Y": y})

        # 骨格ラインの描画
        mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # **手のランドマークを描画**
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            for idx, landmark in enumerate(hand_landmarks.landmark):
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)  # 青色で描画
                landmarks_list.append({"Point": f"Hand_{idx}", "X": x, "Y": y})

            # 手の関節ラインを描画
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # **Streamlitで画像を表示**
    st.image(frame, channels="BGR", use_column_width=True)

    # **座標データを表で表示**
    if landmarks_list:
        df = pd.DataFrame(landmarks_list)
        st.table(df)
