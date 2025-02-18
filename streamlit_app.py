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
    入力画像（NumPy配列）に対して MediaPipe のポーズ検出を行い、
    骨格を描画した画像と検出結果を返す関数。
    """
    results = pose.process(frame)
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
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

# モード選択（リアルタイム と 画像アップロード）
mode = st.sidebar.radio("モードを選択してください", ["リアルタイム", "画像アップロード"])

# 表示モードの切替（座標の表示 / 角度の表示）
display_mode = st.sidebar.radio("表示モードを選択してください", ["座標の表示", "角度の表示"])

# セッションステートで、頂点とその他の点の情報を保持
if "vertex" not in st.session_state:
    st.session_state.vertex = None  # 頂点は1点のみ
if "others" not in st.session_state:
    st.session_state.others = []    # その他の点（リスト）

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
                # 全てのランドマーク情報を作成
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
                
                # Plotly でインタラクティブなグラフ（画像サイズ固定）
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
                
                # 選択されたポイントを「頂点」として追加するボタン
                if st.button("頂点として追加"):
                    if events:
                        clicked_point = events[0]
                        pt_index = clicked_point.get("pointNumber")
                        if pt_index is not None:
                            st.session_state.vertex = pt_index
                            st.success(f"頂点としてポイント {pt_index} を選択しました。")
                    else:
                        st.warning("クリックイベントが検出されませんでした。")
                
                # 選択されたポイントを「その他の点」として追加するボタン
                if st.button("その他の点として追加"):
                    if events:
                        clicked_point = events[0]
                        pt_index = clicked_point.get("pointNumber")
                        if pt_index is not None:
                            if pt_index == st.session_state.vertex:
                                st.info(f"ポイント {pt_index} は既に頂点として選択されています。")
                            elif pt_index not in st.session_state.others:
                                st.session_state.others.append(pt_index)
                                st.success(f"その他の点としてポイント {pt_index} を選択しました。")
                            else:
                                st.info(f"ポイント {pt_index} は既にその他の点として選択されています。")
                    else:
                        st.warning("クリックイベントが検出されませんでした。")
                
                # 現在選択されたポイントを表示
                st.write("【現在の選択状況】")
                if st.session_state.vertex is not None:
                    st.write(f"頂点: ポイント {st.session_state.vertex}")
                else:
                    st.write("頂点: 未選択")
                if st.session_state.others:
                    st.write(f"その他の点: {['ポイント ' + str(pt) for pt in st.session_state.others]}")
                else:
                    st.write("その他の点: 未選択")
                
                # 表示ボタンを押すと、モードに応じた情報を表示
                if st.button("表示"):
                    if display_mode == "座標の表示":
                        display_info = []
                        if st.session_state.vertex is not None:
                            display_info.append(landmark_info[st.session_state.vertex])
                        for pt in st.session_state.others:
                            display_info.append(landmark_info[pt])
                        st.write("【選択されたポイントの座標】")
                        st.dataframe(display_info)
                    elif display_mode == "角度の表示":
                        # 角度計算は、頂点が選択され、かつその他の点から最低2点選択されている場合に行う
                        if st.session_state.vertex is None:
                            st.info("角度を計算するには、まず頂点を選択してください。")
                        elif len(st.session_state.others) < 2:
                            st.info("角度を計算するには、頂点以外から2点以上を選択してください。")
                        else:
                            # ここでは、その他の点のうち最初の2点を使用
                            A = (landmark_info[st.session_state.others][0]["x (abs)"], landmark_info[st.session_state.others][0]["y (abs)"]) if isinstance(st.session_state.others, list) and len(st.session_state.others) > 0 else None
                            # ※ 上記の記述はあくまで例です。より明示的に最初の2点を取得します。
                            pt1 = st.session_state.others[0]
                            pt2 = st.session_state.others[1]
                            A = (landmark_info[pt1]["x (abs)"], landmark_info[pt1]["y (abs)"])
                            B = (landmark_info[st.session_state.vertex]["x (abs)"], landmark_info[st.session_state.vertex]["y (abs)"])
                            C = (landmark_info[pt2]["x (abs)"], landmark_info[pt2]["y (abs)"])
                            angle = compute_angle(A, B, C)
                            if angle is not None:
                                st.success(f"頂点（ポイント {st.session_state.vertex}）での角度: {angle:.2f}°")
                            else:
                                st.error("角度を計算できませんでした。")
            else:
                st.write("関節が検出されませんでした。")
