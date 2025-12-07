import os
import json
import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import joblib
import tempfile
import time
from scipy.spatial import distance as dist
from PIL import Image
from pathlib import Path

# WebRTC
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import av

# --- 1. 全域資源與設定 ---
st.set_page_config(page_title="駕駛疲勞偵測系統", layout="wide", page_icon="🚗")

# MediaPipe 關鍵點索引
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH = [61, 291, 0, 17, 37, 84, 181, 314, 405, 146, 78, 191, 80, 81, 82, 13, 312, 311, 310, 415]
POSE_POINTS = [33, 263, 1, 61, 291, 199]

@st.cache_resource
def load_model():
    model_path = Path(__file__).resolve().parent / "drowsiness_model.pkl"
    return joblib.load(model_path)

@st.cache_resource
def init_mediapipe():
    return mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

try:
    model = load_model()
    face_mesh = init_mediapipe()
    st.sidebar.success("✅ 模型與 MediaPipe 已載入")
except FileNotFoundError:
    st.error("❌ 找不到 `drowsiness_model.pkl`，請先執行訓練程式！")
    st.stop()

# --- 2. 核心運算函式 (無變動) ---

def calculate_ear(points: np.ndarray) -> float:
    A = dist.euclidean(points[1], points[5])
    B = dist.euclidean(points[2], points[4])
    C = dist.euclidean(points[0], points[3])
    return (A + B) / (2.0 * C) if C > 0 else 0.0

def calculate_mar(points: np.ndarray) -> float:
    A = dist.euclidean(points[2], points[10])
    B = dist.euclidean(points[4], points[8])
    C = dist.euclidean(points[0], points[6])
    return (A + B) / (2.0 * C) if C > 0 else 0.0

def extract_features_from_frame(image, face_mesh_instance):
    h, w, _ = image.shape
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh_instance.process(rgb)

    if not results.multi_face_landmarks:
        return None, None

    face_landmarks = results.multi_face_landmarks[0]
    pts = np.array([(int(p.x * w), int(p.y * h)) for p in face_landmarks.landmark])

    left_ear = calculate_ear(pts[LEFT_EYE])
    right_ear = calculate_ear(pts[RIGHT_EYE])
    mar = calculate_mar(pts[MOUTH])

    face_3d = np.array([pts[i] for i in POSE_POINTS], dtype=np.float64)
    face_2d = np.array([pts[i] for i in POSE_POINTS], dtype=np.float64)

    cam_matrix = np.array([[w, 0, h / 2], [0, w, w / 2], [0, 0, 1]], dtype=np.float64)
    dist_matrix = np.zeros((4, 1), dtype=np.float64)

    try:
        success, rot_vec, _ = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
        if not success: return None, pts
        
        rmat, _ = cv2.Rodrigues(rot_vec)
        angles, *_ = cv2.RQDecomp3x3(rmat)
        return [left_ear, right_ear, mar, angles[0] * 360, angles[1] * 360, angles[2] * 360], pts
    except Exception:
        return None, pts

# --- 3. WebRTC 影像處理核心 ---

class DrowsinessTransformer(VideoTransformerBase):
    def __init__(self):
        self.drowsy_count = 0
        self.closed_start = None
        self.face_mesh = init_mediapipe()
        self.model = load_model()

    def transform(self, frame: av.VideoFrame) -> np.ndarray:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1) # 鏡像翻轉，讓畫面更自然
        
        features, _ = extract_features_from_frame(img, self.face_mesh)
        
        alert_text = ""
        model_alert = False
        ear_alert = False

        if features:
            input_data = np.array([features])
            prediction = self.model.predict(input_data)[0]
            prob = self.model.predict_proba(input_data)[0]

            # 模型判定累積
            if prediction == 1 and prob[1] >= ALERT_PROB_THRESHOLD:
                self.drowsy_count += 1
            else:
                self.drowsy_count = 0
            
            model_alert = self.drowsy_count >= CONSEC_FRAMES

            # 閉眼時間累積
            ear_avg = (features[0] + features[1]) / 2.0
            now = time.time()
            if ear_avg < EAR_THRESHOLD:
                if self.closed_start is None: self.closed_start = now
            else:
                self.closed_start = None

            closed_duration = (now - self.closed_start) if self.closed_start else 0.0
            ear_alert = closed_duration >= CLOSED_SECONDS
            
            # 在影像上繪製資訊
            cv2.putText(img, f"Drowsy Prob: {prob[1]:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(img, f"EAR Avg: {ear_avg:.3f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(img, f"Eye Closed: {closed_duration:.1f}s", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if model_alert or ear_alert:
            # 繪製紅色警示框與文字
            h, w, _ = img.shape
            overlay = img.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 255), -1)
            img = cv2.addWeighted(overlay, 0.4, img, 0.6, 0)
            
            alert_msg = "!! WARNING: DROWSY !!"
            text_size, _ = cv2.getTextSize(alert_msg, cv2.FONT_HERSHEY_TRIPLEX, 2, 3)
            text_x = (w - text_size[0]) // 2
            text_y = (h + text_size[1]) // 2
            cv2.putText(img, alert_msg, (text_x, text_y), cv2.FONT_HERSHEY_TRIPLEX, 2, (255, 255, 255), 3)

        return img

# --- 4. Streamlit UI ---

st.title("🚗 駕駛疲勞偵測系統 (WebRTC Demo)")

with st.sidebar:
    st.header("參數設定")
    ALERT_PROB_THRESHOLD = st.slider("模型警示閾值 (prob)", 0.05, 0.9, 0.3, 0.01)
    CONSEC_FRAMES = st.slider("模型連續幀數", 1, 10, 2)
    EAR_THRESHOLD = st.slider("閉眼 EAR 閾值", 0.05, 0.4, 0.2, 0.01)
    CLOSED_SECONDS = st.slider("閉眼判定秒數", 1.0, 5.0, 2.0, 0.5)
    
    st.markdown("---")
    mode = st.radio("選擇模式", ["📷 即時攝影機 (Webcam)", "📂 上傳影片檔案"])

if mode == "📷 即時攝影機 (Webcam)":
    st.markdown("請點擊下方 **START** 按鈕啟動攝影機並開始偵測。")
    st.warning(
        "⚠ 若出現『Connection is taking longer than expected...』，\n"
        "通常是因為目前的網路或防火牆阻擋了 WebRTC 視訊串流。\n"
        "建議改用：\n"
        "1) 手機 4G/5G 開啟本系統連結，或\n"
        "2) 改用『📂 上傳影片檔案』模式。"
    )
    webrtc_streamer(
        key="drowsiness-detection",
        video_processor_factory=DrowsinessTransformer,
        rtc_configuration={
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]},
                {"urls": ["stun:stun1.l.google.com:19302"]},
                {"urls": ["stun:stun2.l.google.com:19302"]},
                {"urls": ["stun:stun.stunprotocol.org:3478"]},
            ]
        },
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

elif mode == "📂 上傳影片檔案":
    st.info("影片上傳模式維持原有邏輯，警示會以文字顯示在下方。")
    uploaded_file = st.file_uploader("上傳影片 (.mp4, .avi, .mov)", type=["mp4", "avi", "mov"])
    
    st_frame = st.image([])
    
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            # --- 影片處理邏輯 (簡化版) ---
            frame = cv2.resize(frame, (640, 480))
            features, _ = extract_features_from_frame(frame, face_mesh)
            
            if features:
                # 這裡可以加入與 WebRTC 相似的警示邏輯，但為求簡潔，僅顯示文字
                input_data = np.array([features])
                prob = model.predict(input_data)[0]
                if prob == 1:
                     st.warning("偵測到疲勞狀態！")
                else:
                     st.success("狀態正常")
            
            st_frame.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        cap.release()
        os.remove(tfile.name)
