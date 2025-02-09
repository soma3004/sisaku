import streamlit as st
import mediapipe as mp
import numpy as np
import cv2
from PIL import Image
import pandas as pd

# Mediapipeのモジュールを初期化
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Mediapipeのインスタンス
pose = mp_pose.Pose()
hands = mp_hands.Hands()
face_mesh = mp_face_mesh.FaceMesh()

# **サイドバーでモード選択**
mode = st.sidebar.radio("🔍 検出モードを選択", ["体の関節のみ", "手の関節のみ", "表情のみ", "すべて"])

# **体の関節のみのリスト（顔は除外）**
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

st.title("📌 Pose, Hand, and Face Detection")

# カメラ画像を取得
img_file = st.camera_input("📷 Take a picture")

if img_file is not None:
    # 画像をPILからOpenCVの形式に変換
    image = Image.open(img_file)
    frame = np.array(image)

    # OpenCVはBGR、StreamlitはRGBを扱うため変換
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Mediapipeで処理
    results_pose = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) if mode in ["体の関節のみ", "すべて"] else None
    results_hands = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) if mode in ["手の関節のみ", "すべて"] else None
    results_face = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) if mode in ["表情のみ", "すべて"] else None

    # 座標データを格納するリスト
    landmarks_list = []

    # **体の関節を描画**
    if results_pose and results_pose.pose_landmarks:
        for landmark_id in BODY_LANDMARKS:
            landmark = results_pose.pose_landmarks.landmark[landmark_id]
            x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)  # 緑色で描画
            landmarks_list.append({"Type": "Body", "Point": landmark_id.name, "X": x, "Y": y})

        # 骨格ラインを描画
        mp_drawing.draw_landmarks(frame, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # **手の関節を描画**
    if results_hands and results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:
            for idx, landmark in enumerate(hand_landmarks.landmark):
                x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)  # 青色で描画
                landmarks_list.append({"Type": "Hand", "Point": f"Hand_{idx}", "X": x, "Y": y})

            # 手の関節ラインを描画
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # **顔のランドマークを描画**
    if results_face and results_face.multi_face_landmarks:
        for face_landmarks in results_face.multi_face_landmarks:
            for idx, landmark in enumerate(face_landmarks.landmark):
                x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)  # 赤色で描画
                landmarks_list.append({"Type": "Face", "Point": f"Face_{idx}", "X": x, "Y": y})

            # 顔のランドマークラインを描画
            mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION)

    # **Streamlitで画像を表示**
    st.image(frame, channels="BGR", use_column_width=True)

    # **座標データを表で表示**
    if landmarks_list:
        df = pd.DataFrame(landmarks_list)
        st.table(df)
