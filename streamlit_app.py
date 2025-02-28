import streamlit as st
import mediapipe as mp
import numpy as np
from PIL import Image
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events
import math

# サイドバーで各種選択
detection_type = st.sidebar.selectbox("検出対象を選択してください", ["骨格検出", "手検出"])
mode = st.sidebar.radio("モードを選択してください", ["リアルタイム", "画像アップロード"])

st.write(f"【検出対象】: {detection_type}")
st.write(f"【モード】: {mode}")

def compute_angle(A, B, C):
    try:
        BA = (A[0] - B[0], A[1] - B[1])
        BC = (C[0] - B[0], C[1] - B[1])
        dot_product = BA[0] * BC[0] + BA[1] * BC[1]
        norm_BA = math.sqrt(BA[0] ** 2 + BA[1] ** 2)
        norm_BC = math.sqrt(BC[0] ** 2 + BC[1] ** 2)
        
        if norm_BA * norm_BC == 0:
            return None
        
        cos_angle = dot_product / (norm_BA * norm_BC)
        
        # acosの範囲外を防ぐ
        if cos_angle < -1.0 or cos_angle > 1.0:
            return None

        angle_rad = math.acos(cos_angle)
        return math.degrees(angle_rad)
    except Exception as e:
        st.error(f"角度計算中にエラーが発生しました: {e}")
        return None

# Pose検出用関数​

def process_frame_pose(frame, pose_detector):
    results = pose_detector.process(frame)
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=4)
        )
    return frame, results


# Hands検出用関数
def process_frame_hands(frame, hands_detector):
    results = hands_detector.process(frame)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=2)
            )
    return frame, results

# セッションステートの初期化（頂点・その他の点）
if "vertex" not in st.session_state:
    st.session_state.vertex = None
if "others" not in st.session_state:
    st.session_state.others = []

# インスタンスの準備
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

# 共通処理：画像（またはカメラ入力）の取得と、オリジナルサイズの散布図作成
def process_and_display(frame, orig_w, orig_h):
    if detection_type == "骨格検出":
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as detector:
            processed_frame, results = process_frame_pose(frame.copy(), detector)
        landmark_info=[]
        x_coords, y_coords, text_coords = [], [], []
        if results.pose_landmarks:
            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                abs_x = int(landmark.x * orig_w)
                abs_y = int(landmark.y * orig_h)
                landmark_info.append({
                    "type": "pose",
                    "id": idx,
                    "x (abs)": abs_x,
                    "y (abs)": abs_y,
                    "x (normalized)": round(landmark.x, 3),
                    "y (normalized)": round(landmark.y, 3),
                    "visibility": round(landmark.visibility, 3)
                })
                x_coords.append(abs_x)
                y_coords.append(abs_y)
                text_coords.append(f"Pose {idx}<br>x:{abs_x}<br>y:{abs_y}")
    else:

        with mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                            min_detection_confidence=0.5, min_tracking_confidence=0.5) as detector:
            processed_frame, results = process_frame_hands(frame.copy(), detector)


        landmark_info = []
        x_coords, y_coords, text_coords = [], [], []
        if results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                for j, landmark in enumerate(hand_landmarks.landmark):
                    abs_x = int(landmark.x * orig_w)
                    abs_y = int(landmark.y * orig_h)
                    landmark_info.append({
                        "type": "hand",
                        "id": f"{hand_idx}_{j}",
                        "x (abs)": abs_x,
                        "y (abs)": abs_y,
                        "x (normalized)": round(landmark.x, 3),
                        "y (normalized)": round(landmark.y, 3),
                        "visibility": None
                    })
                    x_coords.append(abs_x)
                    y_coords.append(abs_y)
                    text_coords.append(f"Hand {hand_idx}_{j}<br>x:{abs_x}<br>y:{abs_y}")
    # 作成した processed_frame とランドマークデータを用いて散布図を作成（オリジナルサイズ）
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
    scale_factor = 0.4
    fig.update_layout(
        width=int(orig_w * scale_factor),
        height=int(orig_h * scale_factor),
        xaxis=dict(fixedrange=True, range=[0, orig_w],scaleanchor="y"),
        yaxis=dict(fixedrange=True, autorange='reversed', range=[0, orig_h]),
        margin=dict(l=0, r=0, t=0, b=0),
        dragmode=False
    )
    return processed_frame, landmark_info, fig

st.title("Hone")

if mode == "画像アップロード":
    uploaded_file = st.file_uploader("画像をアップロード", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        orig_w, orig_h = image.size
        frame = np.array(image)

       
        processed_frame, landmark_info, fig = process_and_display(frame, orig_w, orig_h)
        #koko
        # 画面幅の 4分の3 にするためにカラムを作成
        col1, col2, col3 = st.columns([1, 3, 1])  # 左右の空白 : 中央 (75%)

        with col2:  # 中央の 75% のカラムにプロットを表示
            st.plotly_chart(fig, use_container_width=True)

        
        st.write("【画像上のポイントをクリックして選択してください】")
        events = plotly_events(fig, click_event=True, hover_event=False)
        st.markdown("<br><br>", unsafe_allow_html=True)  # 2行分の余白
        st.markdown("<br><br>", unsafe_allow_html=True)  # 2行分の余白
        st.markdown("<br><br>", unsafe_allow_html=True)  # 2行分の余白
        st.markdown("<br><br>", unsafe_allow_html=True)  # 2行分の余白
        
        # 頂点
        col3, col4 = st.columns(2)
        with col3:
            if st.button("頂点として追加"):
                if events:
                    clicked_point = events[0]
                    pt_index = clicked_point.get("pointNumber")
                    if pt_index is not None:
                        st.session_state.vertex = pt_index
                        st.success(f"頂点としてポイント {pt_index} を選択しました。")
                else:
                    st.warning("クリックイベントが検出されませんでした。")
        with col4:
            if st.button("頂点の選択を取り消す"):
                if st.session_state.vertex is not None:
                    st.session_state.vertex = None
                    st.success("頂点の選択を取り消しました。")
                else:
                    st.info("頂点は未選択です。")
            
        # その他の２点
        col5, col6 = st.columns(2)
        with col5:
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
        
           
        with col6:
            if st.button("最後のその他の点の選択を取り消す"):
                if st.session_state.others:
                    removed = st.session_state.others.pop()
                    st.success(f"その他の点として選択したポイント {removed} の選択を取り消しました。")
                else:
                    st.info("その他の点は未選択です。")
        
        # 現在の選択状況の表示
        st.write("【現在の選択状況】")
        if st.session_state.vertex is not None:
            st.write(f"頂点: ポイント {st.session_state.vertex}")
        else:
            st.write("頂点: 未選択")
        if st.session_state.others:
            st.write(f"その他の点: {['ポイント ' + str(pt) for pt in st.session_state.others]}")
        else:
            st.write("その他の点: 未選択")
        
        # 表示ボタンによる結果表示
        if st.button("表示"):
            if st.session_state.vertex is None:
                st.info("角度を計算するには、まず頂点を選択してください。")
            elif len(st.session_state.others) < 2:
                st.info("角度を計算するには、頂点以外から2点以上選択してください。")
            else:
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
                        
elif mode == "リアルタイム":
    camera_input = st.camera_input("カメラ映像を使用", key="camera")
    if camera_input is not None:
        image = Image.open(camera_input).convert("RGB")
        orig_w, orig_h = image.size
        frame = np.array(image)
        # リアルタイムの場合もオリジナル画像を1/4サイズに縮小して左右表示
        small_image = image.resize((orig_w // 3, orig_h // 3))
        if detection_type == "骨格検出":
            with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as detector:
                processed_frame, results = process_frame_pose(frame.copy(), detector)


            landmark_info = []
            x_coords, y_coords, text_coords = [], [], []
            if results.pose_landmarks:
                for idx, landmark in enumerate(results.pose_landmarks.landmark):
                    abs_x = int(landmark.x * orig_w)
                    abs_y = int(landmark.y * orig_h)
                    landmark_info.append({
                        "type": "pose",
                        "id": idx,
                        "x (abs)": abs_x,
                        "y (abs)": abs_y,
                        "x (normalized)": round(landmark.x, 3),
                        "y (normalized)": round(landmark.y, 3),
                        "visibility": round(landmark.visibility, 3)
                    })
                    x_coords.append(abs_x)
                    y_coords.append(abs_y)
                    text_coords.append(f"Pose {idx}<br>x:{abs_x}<br>y:{abs_y}")
        else:
            with mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                                min_detection_confidence=0.5, min_tracking_confidence=0.5) as detector:
                processed_frame, results = process_frame_hands(frame.copy(), detector)


            landmark_info = []
            x_coords, y_coords, text_coords = [], [], []
            if results.multi_hand_landmarks:
                for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    for j, landmark in enumerate(hand_landmarks.landmark):
                        abs_x = int(landmark.x * orig_w)
                        abs_y = int(landmark.y * orig_h)
                        landmark_info.append({
                            "type": "hand",
                            "id": f"{hand_idx}_{j}",
                            "x (abs)": abs_x,
                            "y (abs)": abs_y,
                            "x (normalized)": round(landmark.x, 3),
                            "y (normalized)": round(landmark.y, 3),
                            "visibility": None
                        })
                        x_coords.append(abs_x)
                        y_coords.append(abs_y)
                        text_coords.append(f"Hand {hand_idx}_{j}<br>x:{abs_x}<br>y:{abs_y}")
        
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
            width=orig_w,
            height=orig_h,
            xaxis=dict(fixedrange=True, range=[0, orig_w]),
            yaxis=dict(fixedrange=True, autorange='reversed', range=[0, orig_h]),
            margin=dict(l=0, r=0, t=0, b=0),
            dragmode=False
        )
        st.plotly_chart(fig, use_container_width=False)
        st.write("【画像上のポイントをクリックして選択してください】")
        events = plotly_events(fig, click_event=True, hover_event=False)
        st.markdown("<br><br>", unsafe_allow_html=True)  # 2行分の余白
        col3, col4 = st.columns(2)
        with col3:
            if st.button("頂点として追加 (リアルタイム)"):
                if events:
                    clicked_point = events[0]
                    pt_index = clicked_point.get("pointNumber")
                    if pt_index is not None:
                        st.session_state.vertex = pt_index
                        st.success(f"頂点としてポイント {pt_index} を選択しました。")
                else:
                    st.warning("クリックイベントが検出されませんでした。")
        with col4:
            if st.button("頂点の選択を取り消す (リアルタイム)"):
                if st.session_state.vertex is not None:
                    st.session_state.vertex = None
                    st.success("頂点の選択を取り消しました。")
                else:
                    st.info("頂点は未選択です。")
            
        col5, col6 = st.columns(2)
        with col5: 
            if st.button("その他の点として追加 (リアルタイム)"):
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
        with col6:
            if st.button("最後のその他の点の選択を取り消す (リアルタイム)"):
                if st.session_state.others:
                    removed = st.session_state.others.pop()
                    st.success(f"その他の点として選択したポイント {removed} の選択を取り消しました。")
                else:
                    st.info("その他の点は未選択です。")
        st.write("【現在の選択状況】")
        if st.session_state.vertex is not None:
            st.write(f"頂点: ポイント {st.session_state.vertex}")
        else:
            st.write("頂点: 未選択")
        if st.session_state.others:
            st.write(f"その他の点: {['ポイント ' + str(pt) for pt in st.session_state.others]}")
        else:
            st.write("その他の点: 未選択")
        if st.button("表示 (リアルタイム)"):
            if st.session_state.vertex is None:
                st.info("角度を計算するには、まず頂点を選択してください。")
            elif len(st.session_state.others) < 2:
                st.info("角度を計算するには、頂点以外から2点以上選択してください。")
            else:
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
        st.write("モード：画像アップロード または リアルタイム を選択してください。")
