import streamlit as st
import mediapipe as mp
import numpy as np
from PIL import Image
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events
import math

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
            # ランドマーク（ポイント）の色を青、サイズを小さく設定
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
            # ポイントをつなぐ線の色を黒、線を太く設定
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=4)
        )
    return frame, results

def compute_angle(A, B, C):
    """
    3点 A, B, C が与えられた場合、B を頂点としたときの角度（度）を計算する。
    A, B, C は (x, y) のタプル。
    """
    BA = (A[0] - B[0], A[1] - B[1])
    BC = (C[0] - B[0], C[1] - B[1])
    dot_product = BA[0] * BC[0] + BA[1] * BC[1]
    norm_BA = math.sqrt(BA[0] ** 2 + BA[1] ** 2)
    norm_BC = math.sqrt(BC[0] ** 2 + BC[1] ** 2)
    if norm_BA * norm_BC == 0:
        return None
    angle_rad = math.acos(dot_product / (norm_BA * norm_BC))
    angle_deg = math.degrees(angle_rad)
    return angle_deg

st.title("骨格検出アプリ")

# モードの切り替え（リアルタイムと画像アップロード）
mode = st.sidebar.radio("モードを選択してください", ["リアルタイム", "画像アップロード"])

# 表示モードの切り替え（座標の表示 / 角度の表示）
display_mode = st.sidebar.radio("表示モードを選択してください", ["座標の表示", "角度の表示"])

# セッションステートに選択済みポイントのリストを保持
if "selected_points" not in st.session_state:
    st.session_state.selected_points = []

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    if mode == "リアルタイム":
        camera_input = st.camera_input("カメラ映像を使用", key="camera")
        if camera_input is not None:
            image = Image.open(camera_input).convert("RGB")
            frame = np.array(image)
            processed_frame, _ = process_frame(frame.copy(), pose)
            st.image(processed_frame, channels="RGB", use_column_width=True, caption="骨格検出結果 (リアルタイム)")
    elif mode == "画像アップロード":
        uploaded_file = st.file_uploader("画像をアップロード", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            frame = np.array(image)
            processed_frame, results = process_frame(frame.copy(), pose)
            st.image(processed_frame, channels="RGB", use_column_width=True, caption="骨格検出結果 (アップロード画像)")
            
            if results.pose_landmarks:
                h, w, _ = frame.shape
                # 全てのランドマークの情報を作成
                landmark_info = []
                x_coords = []
                y_coords = []
                text_coords = []
                for idx, landmark in enumerate(results.pose_landmarks.landmark):
                    abs_x = int(landmark.x * w)
                    abs_y = int(landmark.y * h)
                    landmark_info.append({
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
                
                # Plotly でインタラクティブなグラフを作成（画像サイズに固定）
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
                    xaxis=dict(fixedrange=True, range=[0, w]),
                    yaxis=dict(fixedrange=True, autorange='reversed', range=[0, h]),
                    margin=dict(l=0, r=0, t=0, b=0),
                    dragmode=False
                )
                st.write("【画像上のポイントをクリックして選択してください】")
                events = plotly_events(fig, click_event=True, hover_event=False)
                
                # 選択ボタンを押すと、直近のクリックイベントからポイントを追加
                if st.button("選択を追加"):
                    if events:
                        clicked_point = events[0]
                        pt_index = clicked_point.get("pointNumber")
                        if pt_index is not None:
                            if pt_index not in st.session_state.selected_points:
                                st.session_state.selected_points.append(pt_index)
                                st.success(f"ポイント {pt_index} を選択しました。")
                            else:
                                st.info(f"ポイント {pt_index} は既に選択されています。")
                    else:
                        st.warning("クリックイベントが検出されませんでした。")
                
                # 選択されたポイントを表示
                selected_str = [f"ポイント {pt}" for pt in st.session_state.selected_points]
                st.write("【選択済みポイント】")
                selected_choice = st.multiselect("選択済みポイント:", options=selected_str, default=selected_str)
                
                # 表示ボタンを押すと、モードに応じた情報を表示
                if st.button("表示"):
                    # 選択されたポイントのインデックスを取得
                    selected_indices = [int(s.split()[1]) for s in selected_choice]
                    if display_mode == "座標の表示":
                        # 座標情報のテーブルを表示
                        display_info = [landmark_info[i] for i in selected_indices]
                        st.write("【選択されたポイントの座標】")
                        st.dataframe(display_info)
                    elif display_mode == "角度の表示":
                        if len(selected_indices) == 3:
                            A = (landmark_info[selected_indices[0]]["x (abs)"], landmark_info[selected_indices[0]]["y (abs)"])
                            B = (landmark_info[selected_indices[1]]["x (abs)"], landmark_info[selected_indices[1]]["y (abs)"])
                            C = (landmark_info[selected_indices[2]]["x (abs)"], landmark_info[selected_indices[2]]["y (abs)"])
                            angle = compute_angle(A, B, C)
                            if angle is not None:
                                st.success(f"頂点（ポイント {selected_indices[1]}）での角度: {angle:.2f}°")
                            else:
                                st.error("角度を計算できませんでした。")
                        else:
                            st.info("角度を計算するには、3つのポイントを選択してください。")
            else:
                st.write("関節が検出されませんでした。")
