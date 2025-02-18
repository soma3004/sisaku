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
            # すべてのポイントを描画（この描画は変更しなくてもよい）
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=4)
        )
    return frame, results

st.title("骨格検出アプリ")

# サイドバーでモードの切り替え
mode = st.sidebar.radio("モードを選択してください", ["リアルタイム", "画像アップロード"])

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    if mode == "リアルタイム":
        camera_input = st.camera_input("カメラ映像を使用", key="camera")
        if camera_input is not None:
            image = Image.open(camera_input).convert("RGB")
            frame = np.array(image)
            processed_frame, results = process_frame(frame.copy(), pose)
            st.image(processed_frame, channels="RGB", use_column_width=True, caption="骨格検出結果 (リアルタイム)")
    elif mode == "画像アップロード":
        uploaded_file = st.file_uploader("画像をアップロード", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            frame = np.array(image)
            processed_frame, results = process_frame(frame.copy(), pose)
            st.image(processed_frame, channels="RGB", use_column_width=True, caption="骨格検出結果 (アップロード画像)")
            
            if results.pose_landmarks:
                # 表示するランドマークのインデックスを指定（例として主要なポイントのみ）
                selected_indices = [0, 11, 12, 23, 24]
                landmark_coords = []
                x_coords = []
                y_coords = []
                text_coords = []
                h, w, _ = frame.shape
                for idx, landmark in enumerate(results.pose_landmarks.landmark):
                    if idx in selected_indices:
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
                        x_coords.append(abs_x)
                        y_coords.append(abs_y)
                        text_coords.append(f"ID: {idx}<br>x: {abs_x}<br>y: {abs_y}")
                
                # Plotly によるインタラクティブな表示（選択したポイントのみを表示）
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
                fig.update_layout(
                    yaxis=dict(autorange='reversed'),
                    margin=dict(l=0, r=0, t=0, b=0)
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # テーブルは選択したポイントのみ表示
                st.write("選択した関節のポイントの座標:")
                st.dataframe(landmark_coords)
            else:
                st.write("関節が検出されませんでした。")
