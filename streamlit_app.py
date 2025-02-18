import streamlit as st
import mediapipe as mp
import numpy as np
from PIL import Image

# MediaPipe の準備
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def process_frame(frame, pose):
    """
    入力画像（NumPy配列）に対してMediaPipeのポーズ検出を行い、
    骨格を描画した画像と検出結果を返す関数。
    """
    results = pose.process(frame)
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            # ポイントの色を青色、サイズを小さく (circle_radius=2) 設定
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
            # 接続線の色を黒色、線を太く (thickness=4) 設定
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=4)
        )
    return frame, results

st.title("骨格検出アプリ")

# サイドバーでモードの切り替え
mode = st.sidebar.radio("モードを選択してください", ["リアルタイム", "画像アップロード"])

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    if mode == "リアルタイム":
        # カメラ入力
        camera_input = st.camera_input("カメラ映像を使用", key="camera")
        if camera_input is not None:
            image = Image.open(camera_input).convert("RGB")
            frame = np.array(image)
            processed_frame, _ = process_frame(frame.copy(), pose)
            st.image(processed_frame, channels="RGB", use_column_width=True, caption="骨格検出結果")
    
    elif mode == "画像アップロード":
        # 画像アップロード
        uploaded_file = st.file_uploader("画像をアップロード", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            frame = np.array(image)
            processed_frame, results = process_frame(frame.copy(), pose)
            st.image(processed_frame, channels="RGB", use_column_width=True, caption="骨格検出結果")
            
            # 関節の座標を表示
            if results.pose_landmarks:
                landmark_coords = []
                h, w, _ = frame.shape  # 画像の高さ、幅を取得
                for idx, landmark in enumerate(results.pose_landmarks.landmark):
                    abs_x = int(landmark.x * w)
                    abs_y = int(landmark.y * h)
                    landmark_coords.append({
                        "Landmark": idx,
                        "x (normalized)": round(landmark.x, 3),
                        "y (normalized)": round(landmark.y, 3),
                        "z (normalized)": round(landmark.z, 3),
                        "x (abs)": abs_x,
                        "y (abs)": abs_y,
                        "visibility": round(landmark.visibility, 3)
                    })
                st.write("関節のポイントの座標:")
                st.dataframe(landmark_coords)
            else:
                st.write("関節が検出されませんでした。")
