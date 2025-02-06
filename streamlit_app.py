import streamlit as st
import numpy as np
import cv2
import mediapipe as mp
from PIL import Image

def image_face_detection():
    st.write('顔検出：画像ファイルから顔を検出します')
    img_file = st.file_uploader("画像を選択", type="jpg")
    
    # MediaPipe の顔検出の設定
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils

    if img_file is not None:
        with st.spinner("検出中..."):
            # PILを使って画像を開く
            img = Image.open(img_file)
            
            # PIL画像をOpenCV形式（numpy配列）に変換
            img = np.array(img)
            
            # RGBからBGRに変換（MediaPipeの処理に必要）
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # MediaPipe の顔検出モデルを初期化
            with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
                # 画像を処理し、結果を取得
                results = face_detection.process(img)
                
                if not results.detections:
                    st.write('顔検出に失敗しました')
                else:
                    # 検出結果を描画するために画像をコピー
                    annotated_image = img.copy()
                    
                    # 検出された顔を画像に描画
                    for detection in results.detections:
                        mp_drawing.draw_detection(annotated_image, detection)
                    
                    # BGRからRGBに戻してStreamlitで表示できる形式に変換
                    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

                    # 結果の画像を表示
                    st.image(annotated_image, caption="検出結果")

# Streamlitアプリケーションを実行
if __name__ == '__main__':
    image_face_detection()
