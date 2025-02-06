import mediapipe as mp
import cv2
import numpy as np
import streamlit as st
from PIL import Image
def image_face_detection():
    st.write('face detection from image file')
    img_file = st.file_uploader("画像を選択", type="jpg")
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils

    if img_file is not None:
        with st.spinner("検出中"):
            img = Image.open(img_file)
            img = np.array(img, dtype=np.uint8)

            with mp_face_detection.FaceDetection(model_selection=1, 
                                                 min_detection_confidence=0.5) as face_detection:

                results = face_detection.process(img)
                if not results.detections:
                    st.write('Detection failure')
                else:
                    annotated_image = img.copy()
                    for detection in results.detections:
                        mp_drawing.draw_detection(annotated_image, detection)

                    st.image(annotated_image, caption="検出結果")

def image_hand_detection():
    st.write('hand detection from image file')
    img_file = st.file_uploader("画像を選択", type="jpg")

    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands

    if img_file is not None:
        with st.spinner("検出中"):
            img = Image.open(img_file)
            img = np.array(img, dtype=np.uint8)
            with mp_hands.Hands(static_image_mode=True,
                                max_num_hands=2,
                                min_detection_confidence=0.5) as hands:
                
                results = hands.process(img)
                if not results.multi_hand_landmarks:
                    st.write('Detection failure')
                else:
                    annotated_image = img.copy()
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            annotated_image,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style())
                    
                    st.image(annotated_image, caption="検出結果")
def image_pose_detection():
    st.write('pose detection from image file')
    img_file = st.file_uploader("画像を選択", type="jpg")
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose

    if img_file is not None:
        with st.spinner("検出中"):
            img = Image.open(img_file)
            img = np.array(img, dtype=np.uint8)
            with mp_pose.Pose(static_image_mode=True,
                            model_complexity=2,
                            enable_segmentation=True,
                            min_detection_confidence=0.5) as pose:
                
                results = pose.process(img)
                if not results.pose_landmarks:
                    st.write('Detection failure')
                else:
                    annotated_image = img.copy()
                    mp_drawing.draw_landmarks(annotated_image,
                                            results.pose_landmarks,
                                            mp_pose.POSE_CONNECTIONS,
                                            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

                    st.image(annotated_image, caption="検出結果")
def image_holistic_detection():
    st.write('holistic detection from image file')
    img_file = st.file_uploader("画像を選択", type="jpg")
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_holistic = mp.solutions.holistic

    if img_file is not None:
        with st.spinner("検出中"):
            img = Image.open(img_file)
            img = np.array(img, dtype=np.uint8)
            with mp_holistic.Holistic(static_image_mode=True,
                                    model_complexity=2,
                                    enable_segmentation=True,
                                    refine_face_landmarks=True) as holistic:
                
                results = holistic.process(img)
                if not results.pose_landmarks:
                    st.write('Detection failure')
                else:
                    annotated_image = img.copy()
                    mp_drawing.draw_landmarks(
                        annotated_image,
                        results.face_landmarks,
                        mp_holistic.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                        )
                    mp_drawing.draw_landmarks(
                        annotated_image,
                        results.pose_landmarks,
                        mp_holistic.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
                        )
                    mp_drawing.draw_landmarks(
                        annotated_image,
                        results.left_hand_landmarks,
                        mp_holistic.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                        )
                    mp_drawing.draw_landmarks(
                        annotated_image,
                        results.right_hand_landmarks,
                        mp_holistic.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                        )
                    st.image(annotated_image, caption="検出結果")

def main():
    st.title('mediapipe app')

    select_app = st.sidebar.radio('app', ('face detection from image file', 
                                          'hand detection from image file', 
                                          'pose detection from image file',
                                          'holistic detection from image file'))

    func_dict.get(select_app, select_err)()

def select_err():
    st.error('ERROR')


func_dict = {
    'face detection from image file': image_face_detection,
    'hand detection from image file': image_hand_detection,
    'pose detection from image file': image_pose_detection,
    'holistic detection from image file': image_holistic_detection,
}

if __name__ == '__main__':
    main()

