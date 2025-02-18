import streamlit as st
import mediapipe as mp
import numpy as np
from PIL import Image

# MediaPipe の準備
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def process_frame(frame, pose):
    """
    入力のRGB画像（numpy配列）に対してMediaPipeのポーズ検出を実行し、
    検出結果（骨格）を描画した画像を返す関数。
    """
    # パフォーマンス向上のため、一時的に書き込み不可にする
    frame.flags.writeable = False
    results = pose.process(frame)
    frame.flags.writeable = True

    # ポーズ検出に成功した場合、骨格を描画
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
        )
    return frame

# Poseオブジェクトはwithブロックで管理する
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

    st.title("リアルタイムおよび写真骨格検出")

    # カメラ入力（写真として取得）
    camera_input = st.camera_input("カメラ映像を使用", key="camera")

    # 写真アップロード
    uploaded_image = st.file_uploader("写真をアップロード", type=["jpg", "jpeg", "png"])

    if camera_input is not None:
        # 画像をPILで読み込み、RGBに変換
        image = Image.open(camera_input).convert("RGB")
        frame = np.array(image)

        # オリジナル画像を表示
        st.image(frame, channels="RGB", caption="カメラ映像", use_container_width=True)

        # 描画用に元画像のコピーを作成して処理
        processed_frame = process_frame(frame.copy(), pose)

        # 骨格描画後の画像を表示
        st.image(processed_frame, channels="RGB", caption="骨格検出結果", use_container_width=True)

    elif uploaded_image is not None:
        image = Image.open(uploaded_image).convert("RGB")
        frame = np.array(image)

        st.image(frame, channels="RGB", caption="アップロードされた画像", use_container_width=True)

        processed_frame = process_frame(frame.copy(), pose)
        st.image(processed_frame, channels="RGB", caption="骨格検出結果", use_container_width=True)

    else:
        st.warning("カメラ映像または画像をアップロードしてください。")
