import streamlit as st
import mediapipe as mp
import numpy as np
from PIL import Image
import cv2



# Mediapipeのポーズ推定モジュールを初期化
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Streamlitでカメラ映像を表示
st.title("Pose Detection with Coordinates")
st.text("Press 'q' to exit the application.")

# カメラを初期化
cap = cv2.VideoCapture(0)

# 処理ループ
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # BGRからRGBに変換（MediapipeはRGB形式を使用）
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # ポーズ推定を実行
    results = pose.process(rgb_frame)

    # 骨格点を描画
    if results.pose_landmarks:
        for landmark in results.pose_landmarks.landmark:
            # 各関節の座標を取得
            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])

            # 座標を表示
            cv2.putText(frame, f"({x}, {y})", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

            # 各関節点を描画
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    # 結果をStreamlitで表示
    st.image(frame, channels="BGR", use_column_width=True)

    # 'q'キーが押されたら終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

