import streamlit as st
import mediapipe as mp
import numpy as np
from PIL import Image
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events

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

st.title("骨格検出アプリ")

# モードの切り替え（今回は画像アップロードモードのみで実装）
mode = st.sidebar.radio("モードを選択してください", ["リアルタイム", "画像アップロード"])

# セッションステートに選択済みポイントのリストを保持
if "selected_points" not in st.session_state:
    st.session_state.selected_points = []

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    if mode == "リアルタイム":
        # リアルタイムモードでは通常の画像表示のみ
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
                # 全てのランドマークの絶対座標とテキスト情報を作成
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
                
                # Plotly でインタラクティブなグラフを作成
                fig = go.Figure()
                fig.add_trace(go.Image(z=processed_frame))
                scatter = go.Scatter(
                    x=x_coords,
                    y=y_coords,
                    mode="markers",
                    marker=dict(color="blue", size=6),
                    text=text_coords,
                    hoverinfo="text"
                )
                fig.add_trace(scatter)
                fig.update_layout(
                    yaxis=dict(autorange='reversed'),
                    margin=dict(l=0, r=0, t=0, b=0)
                )
                st.write("【ポイントをクリックして選択してください】")
                # plotly_events でクリックイベントを取得
                events = plotly_events(fig, click_event=True, hover_event=False)
                
                # 選択ボタンを押すと、直近のクリックイベントからポイントを追加
                if st.button("選択を追加"):
                    if events:
                        # ここでは最初のイベントのみを採用（複数クリックした場合は工夫可能）
                        clicked_point = events[0]
                        # clicked_point["pointNumber"] は scatter 内のインデックスと一致
                        pt_index = clicked_point.get("pointNumber")
                        if pt_index is not None:
                            if pt_index not in st.session_state.selected_points:
                                st.session_state.selected_points.append(pt_index)
                                st.success(f"ポイント {pt_index} を選択しました。")
                            else:
                                st.info(f"ポイント {pt_index} は既に選択されています。")
                    else:
                        st.warning("クリックイベントが検出されませんでした。")
                
                # 選択されたポイントをセレクトボックスに表示
                selected_str = [f"ポイント {pt}" for pt in st.session_state.selected_points]
                st.write("【選択済みポイント】")
                selected_choice = st.multiselect("選択済みポイント:", options=selected_str, default=selected_str)
                
                # 座標を表示するボタン
                if st.button("座標を表示"):
                    # フィルタ：選択ボックスで選ばれたポイントのインデックスを取得
                    # 例: "ポイント 11" -> 11 と変換
                    selected_indices = [int(s.split()[1]) for s in selected_choice]
                    display_info = [landmark_info[i] for i in selected_indices]
                    st.write("【選択されたポイントの座標】")
                    st.dataframe(display_info)
            else:
                st.write("関節が検出されませんでした。")
