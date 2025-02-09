import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import time

# Mediapipeのセットアップ
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()
hands = mp_hands.Hands()

st.title("📌 骨格検出アプリ")

# **サイドバーでモード選択**
main_mode = st.sidebar.radio("🔍 検出モードを選択", ["リアルタイム検出", "写真から座標を検出", "動画から座標を検出"])
sub_mode = st.sidebar.radio("📌 検出対象を選択", ["体の関節のみ", "手の関節のみ", "すべて"])

# **骨格ポイントに最も近い点を見つける関数**
def find_nearest_landmark(mouse_x, mouse_y, landmarks, frame):
    min_dist = float('inf')
    nearest_point = None
    for idx, landmark in enumerate(landmarks):
        x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
        dist = np.sqrt((mouse_x - x) ** 2 + (mouse_y - y) ** 2)
        if dist < min_dist:
            min_dist = dist
            nearest_point = (x, y, idx)
    return nearest_point if min_dist < 20 else None  # 20px以内なら表示

# **リアルタイム検出モード**
if main_mode == "リアルタイム検出":
    st.subheader("🎥 リアルタイムで骨格を検出中...")
    cap = cv2.VideoCapture(0)
    FRAME_WINDOW = st.empty()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("カメラ映像の取得に失敗しました。")
            break
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

# **写真から座標を検出するモード**
elif main_mode == "写真から座標を検出":
    st.subheader("📸 写真をアップロードしてください")
    img_file = st.file_uploader("画像をアップロード", type=["jpg", "jpeg", "png"])
    if img_file is not None:
        image = np.array(cv2.imdecode(np.frombuffer(img_file.read(), np.uint8), 1))
        results_pose = pose.process(image) if sub_mode in ["体の関節のみ", "すべて"] else None
        results_hands = hands.process(image) if sub_mode in ["手の関節のみ", "すべて"] else None
        
        if results_pose and results_pose.pose_landmarks:
            mp_drawing.draw_landmarks(image, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        if results_hands and results_hands.multi_hand_landmarks:
            for hand_landmarks in results_hands.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        st.image(image, channels="BGR", use_column_width=True)

# **動画から座標を検出するモード**
elif main_mode == "動画から座標を検出":
    st.subheader("🎞️ 動画をアップロードしてください")
    video_file = st.file_uploader("動画をアップロード", type=["mp4", "mov", "avi"])
    if video_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
            temp_video.write(video_file.read())
            temp_video_path = temp_video.name
        cap = cv2.VideoCapture(temp_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        delay = 1 / fps if fps > 0 else 0.03
        FRAME_WINDOW = st.empty()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results_pose = pose.process(frame) if sub_mode in ["体の関節のみ", "すべて"] else None
            results_hands = hands.process(frame) if sub_mode in ["手の関節のみ", "すべて"] else None
            
            if results_pose and results_pose.pose_landmarks:
                mp_drawing.draw_landmarks(frame, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            if results_hands and results_hands.multi_hand_landmarks:
                for hand_landmarks in results_hands.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            FRAME_WINDOW.image(frame, channels="RGB")
            time.sleep(delay)
        cap.release()
        st.success("動画の処理が完了しました 🎉")
