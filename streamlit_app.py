import streamlit as st
import mediapipe as mp
import numpy as np
from PIL import Image
import plotly.graph_objects as go

# MediaPipe の準備
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def process_frame(frame, pose):
    """
    入力画像（NumPy配列）に対して MediaPipe でポーズ検出を行い、
    骨格を描画した画像と検出結果を返す関数。
    """
    results = pose.process(frame)
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            # ポイントの色を青、サイズを小さく設定
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
            # 接続線の色を黒、線を太く設定
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=4)
        )
    return frame, results

st.title("骨格検出アプリ")

# サイドバーでモード切替
mode = st.sidebar.radio("モードを選択してください", ["リアルタイム", "画像アップロード"])

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    if mode == "リアルタイム":
        # カメラ入力
        camera_input = st.camera_input("カメラ映像を使用", key="camera")
        if camera_input is not None:
            image = Image.open(camera_input).convert("RGB")
            frame = np.array(image)
            processed_frame, results = process_frame(frame.copy(), pose)
            
            # まずは通常の画像表示（骨格描画済み）
            st.image(processed_frame, channels="RGB", use_column_width=True, caption="骨格検出結果")
            
            # Plotly でインタラクティブ表示（ホバー時に座標を表示）
            if results.pose_landmarks:
                h, w, _ = frame.shape
                x_coords = []
                y_coords = []
                text_coords = []
                for idx, landmark in enumerate(results.pose_landmarks.landmark):
                    abs_x = int(landmark.x * w)
                    abs_y = int(landmark.y * h)
                    x_coords.append(abs_x)
                    y_coords.append(abs_y)
                    text_coords.append(f"ID: {idx}<br>x: {abs_x}<br>y: {abs_y}")
                
                fig = go.Figure()
                # 背景画像として骨格描画済みの画像を表示
                fig.add_trace(go.Image(z=processed_frame))
                # 各ランドマークを散布図として重ねる
                fig.add_trace(go.Scatter(
                    x=x_coords,
                    y=y_coords,
                    mode="markers",
                    marker=dict(color="blue", size=6),
                    text=text_coords,
                    hoverinfo="text"
                ))
                # 画像上部の座標系が反転しているため、y軸を反転
                fig.update_layout(yaxis=dict(autorange='reversed'),
                                  margin=dict(l=0, r=0, t=0, b=0))
                st.plotly_chart(fig, use_container_width=True)
    
    elif mode == "画像アップロード":
        # 画像アップロード
        uploaded_file = st.file_uploader("画像をアップロード", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            frame = np.array(image)
            processed_frame, results = process_frame(frame.copy(), pose)
            
            st.image(processed_frame, channels="RGB", use_column_width=True, caption="骨格検出結果")
            
            # Plotly によるインタラクティブ表示
            if results.pose_landmarks:
                h, w, _ = frame.shape
                x_coords = []
                y_coords = []
                text_coords = []
                for idx, landmark in enumerate(results.pose_landmarks.landmark):
                    abs_x = int(landmark.x * w)
                    abs_y = int(landmark.y * h)
                    x_coords.append(abs_x)
                    y_coords.append(abs_y)
                    text_coords.append(f"ID: {idx}<br>x: {abs_x}<br>y: {abs_y}")
                
                fig = go.Figure()
                fig.add_trace(go.Image(z=processed_frame))
                fig.add_trace(go.Scatter(
                    x=x_coords,
                    y=y_coords,
                    mode="markers",
                    marker=dict(color="blue", size=6),
                    text=text_coords,
                    hoverinfo="text"
                ))
                fig.update_layout(yaxis=dict(autorange='reversed'),
                                  margin=dict(l=0, r=0, t=0, b=0))
                st.plotly_chart(fig, use_container_width=True)
