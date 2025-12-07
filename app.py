import os
import time
import tempfile
from pathlib import Path

import av
import cv2
import joblib
import mediapipe as mp
import numpy as np
import streamlit as st
from scipy.spatial import distance as dist
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
from PIL import Image, ImageDraw, ImageFont

# -------------------------------------------------
# 1. Streamlit 基本設定
# -------------------------------------------------
st.set_page_config(page_title="駕駛疲勞偵測系統", layout="wide", page_icon="🚗")

# MediaPipe 關鍵點索引
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH = [78, 81, 13, 311, 308, 402, 14, 178]
POSE_POINTS = [33, 263, 1, 61, 291, 199]

# -------------------------------------------------
# 1-1. 模型載入
# -------------------------------------------------
@st.cache_resource
def load_model():
    # 取得 app.py 所在資料夾
    base_dir = Path(__file__).resolve().parent
    # 假設模型放在 models/ 資料夾下
    model_path = base_dir / "models" / "drowsiness_model.pkl"

    if not model_path.exists():
        # 如果找不到，試著找當前目錄 (有些部署環境會攤平)
        model_path = base_dir / "drowsiness_model.pkl"

    if not model_path.exists():
        st.error("❌ 找不到模型檔 (drowsiness_model.pkl)")
        return None

    return joblib.load(model_path)

@st.cache_resource
def init_mediapipe():
    return mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

def draw_alert(image, text):
    """在影像上繪製警報文字和圖示"""
    h, w, _ = image.shape
    
    # 建立一個透明覆蓋層
    overlay = Image.new('RGBA', (w, h), (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)

    # 繪製紅色三角形
    triangle_size = 80
    top_point = (w // 2, h // 2 - triangle_size // 2)
    left_point = (w // 2 - triangle_size // 2, h // 2 + triangle_size // 2)
    right_point = (w // 2 + triangle_size // 2, h // 2 + triangle_size // 2)
    draw.polygon([top_point, left_point, right_point], fill=(255, 0, 0, 180))
    
    # 在三角形內繪製驚嘆號
    try:
        font_path = "arial.ttf"
        font = ImageFont.truetype(font_path, 60)
        draw.text((w // 2 - 10, h // 2 - 20), "!", font=font, fill=(255, 255, 255, 255))
    except IOError:
        draw.text((w // 2 - 10, h // 2 - 20), "!", fill=(255, 255, 255, 255))


    # 繪製警告文字
    try:
        # 嘗試使用支持中文的字體
        font = ImageFont.truetype("msjh.ttc", 30) # Microsoft JhengHei
    except IOError:
        # 若找不到，使用預設字體
        try:
            font = ImageFont.truetype("arial.ttf", 30)
        except IOError:
            font = ImageFont.load_default()
        
    text_bbox = draw.textbbox((0,0), text, font=font)
    text_w = text_bbox[2] - text_bbox[0]
    text_h = text_bbox[3] - text_bbox[1]
    
    text_pos = (w // 2 - text_w // 2, h // 2 + triangle_size // 2 + 20)
    
    # 加個半透明背景
    draw.rectangle(
        (text_pos[0] - 10, text_pos[1] - 10, text_pos[0] + text_w + 10, text_pos[1] + text_h + 10),
        fill=(0, 0, 0, 128)
    )
    draw.text(text_pos, text, font=font, fill=(255, 255, 0, 255))

    # 將覆蓋層合併回原圖
    cv2_image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(cv2_image_rgb)
    pil_image = Image.alpha_composite(pil_image.convert('RGBA'), overlay)
    
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

# -------------------------------------------------
def calculate_ear(eye_points, landmarks):
    p = [landmarks[i] for i in eye_points]
    A = dist.euclidean((p[1].x, p[1].y), (p[5].x, p[5].y))
    B = dist.euclidean((p[2].x, p[2].y), (p[4].x, p[4].y))
    C = dist.euclidean((p[0].x, p[0].y), (p[3].x, p[3].y))
    return (A + B) / (2.0 * C) if C != 0 else 0

def calculate_mar(mouth_points, landmarks):
    p = [landmarks[i] for i in mouth_points]
    A = dist.euclidean((p[1].x, p[1].y), (p[7].x, p[7].y))
    B = dist.euclidean((p[2].x, p[2].y), (p[6].x, p[6].y))
    C = dist.euclidean((p[3].x, p[3].y), (p[5].x, p[5].y))
    D = dist.euclidean((p[0].x, p[0].y), (p[4].x, p[4].y))
    return (A + B + C) / (2.0 * D) if D != 0 else 0

def extract_features(image, face_mesh_instance):
    h, w, _ = image.shape
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh_instance.process(rgb)

    if not results.multi_face_landmarks:
        return None, None

    landmarks = results.multi_face_landmarks[0].landmark
    pts = np.array([(int(p.x * w), int(p.y * h)) for p in landmarks])

    left_ear = calculate_ear(LEFT_EYE, landmarks)
    right_ear = calculate_ear(RIGHT_EYE, landmarks)
    mar = calculate_mar(MOUTH, landmarks)

    # Head Pose
    face_3d = []
    face_2d = []
    for idx in POSE_POINTS:
        lm = landmarks[idx]
        x, y = int(lm.x * w), int(lm.y * h)
        face_2d.append([x, y])
        face_3d.append([x, y, lm.z])
    
    face_2d = np.array(face_2d, dtype=np.float64)
    face_3d = np.array(face_3d, dtype=np.float64)

    cam_matrix = np.array([[w, 0, h / 2], [0, w, w / 2], [0, 0, 1]], dtype=np.float64)
    dist_matrix = np.zeros((4, 1), dtype=np.float64)

    try:
        success, rot_vec, _ = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix, flags=cv2.SOLVEPNP_ITERATIVE)
        if not success: return None, pts
        rmat, _ = cv2.Rodrigues(rot_vec)
        angles, *_ = cv2.RQDecomp3x3(rmat)
        
        pitch = angles[0] * 360
        yaw = angles[1] * 360
        roll = angles[2] * 360

        return [left_ear, right_ear, mar, pitch, yaw, roll], pts
    except Exception:
        return None, pts

# -------------------------------------------------
# 3. 新版 WebRTC Processor
# -------------------------------------------------
# 注意：繼承 VideoProcessorBase
class DrowsinessProcessor(VideoProcessorBase):
    def __init__(self):
        # 參數初始化
        self.use_adaptive_ear = True
        self.calibration_seconds = 10
        self.k_factor = 2.0
        self.ear_threshold = 0.25
        self.closed_seconds = 1.0
        self.alert_prob_threshold = 0.3
        self.consec_frames = 1

        # 狀態變數
        self.drowsy_count = 0
        self.closed_start = None
        
        # 在這裡初始化 MediaPipe 和 模型 (每個連線獨立一份)
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=False, min_detection_confidence=0.5, min_tracking_confidence=0.5
        )
        self.model = load_model()

        # 校準相關
        self.is_calibrating = False
        self.calibration_start_time = None
        self.ear_samples = []
        self.adaptive_ear_threshold = None
        self.needs_calibration_start = True # 自動開始校準

    def start_calibration(self):
        self.is_calibrating = True
        self.calibration_start_time = time.time()
        self.ear_samples = []
        self.adaptive_ear_threshold = None

    # !!! 關鍵修改：舊版叫 transform, 新版叫 recv !!!
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        # 1. 將 av.VideoFrame 轉為 numpy array (BGR)
        img = frame.to_ndarray(format="bgr24")
        h, w, _ = img.shape
        
        # 鏡像翻轉 (可選)
        img = cv2.flip(img, 1)
        
        # 自動開始校準
        if self.use_adaptive_ear and self.needs_calibration_start:
            self.start_calibration()
            self.needs_calibration_start = False

        # 2. 特徵提取
        features, _ = extract_features(img, self.face_mesh)

        # 3. 邏輯處理 (校準或偵測)
        # ------------------------------------------------
        if self.use_adaptive_ear and self.is_calibrating:
            # --- 校準模式 ---
            now = time.time()
            remain = self.calibration_seconds - (now - self.calibration_start_time)

            if remain > 0:
                if features:
                    ear_avg = (features[0] + features[1]) / 2.0
                    self.ear_samples.append(ear_avg)
                
                # 繪製倒數文字（使用 PIL 支援中文）
                img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(img_pil)
                
                try:
                    font_large = ImageFont.truetype("msjh.ttc", 36)
                    font_medium = ImageFont.truetype("msjh.ttc", 24)
                except:
                    font_large = ImageFont.load_default()
                    font_medium = ImageFont.load_default()
                
                draw.text((30, 50), f"校準中... {int(remain)}秒", font=font_large, fill=(255, 255, 0))
                draw.text((30, 100), "請保持眼睛自然睜開並正常眨眼", font=font_medium, fill=(255, 255, 0))
                
                img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            else:
                # 校準結束
                self.is_calibrating = False
                if self.ear_samples:
                    mean_ear = np.mean(self.ear_samples)
                    std_ear = np.std(self.ear_samples)
                    self.adaptive_ear_threshold = mean_ear - self.k_factor * std_ear
                    # 注意：這裡不能用 st.success，因為是在不同執行緒，只能畫在圖上
                    print(f"校準完成: Threshold={self.adaptive_ear_threshold}")

        else:
            # --- 偵測模式 ---
            current_th = self.ear_threshold
            if self.use_adaptive_ear and self.adaptive_ear_threshold is not None:
                current_th = self.adaptive_ear_threshold

            model_alert = False
            ear_alert = False
            
            if self.model is not None and features:
                ear_avg = (features[0] + features[1]) / 2.0
                now = time.time()

                # XGBoost 預測
                # features: [L_EAR, R_EAR, MAR, P, Y, R]
                try:
                    input_feat = np.array([features])
                    pred = self.model.predict(input_feat)[0]
                    prob = self.model.predict_proba(input_feat)[0]
                    
                    # 模型判定邏輯
                    if pred == 1 and prob[1] >= self.alert_prob_threshold:
                        self.drowsy_count += 1
                    else:
                        self.drowsy_count = 0
                    
                    if self.drowsy_count >= self.consec_frames:
                        model_alert = True
                except Exception as e:
                    print(f"Prediction Error: {e}")
                    prob = [0.0, 0.0]  # 錯誤時設定預設值

                # 純 EAR 判定邏輯 (閉眼過久)
                if ear_avg < current_th:
                    if self.closed_start is None:
                        self.closed_start = now
                else:
                    self.closed_start = None

                closed_duration = (now - self.closed_start) if self.closed_start else 0
                if closed_duration >= self.closed_seconds:
                    ear_alert = True

                # 取得 MAR 值用於顯示
                mar_value = features[2]
                
                # 轉換為 PIL 格式以支援中文
                img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(img_pil)
                
                try:
                    font_small = ImageFont.truetype("msjh.ttc", 20)
                except:
                    font_small = ImageFont.load_default()
                
                # 繪製各項數值（英文數字用 OpenCV 較快）
                cv2.putText(img, f"Prob: {prob[1]:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(img, f"EAR: {ear_avg:.2f} (Th:{current_th:.2f})", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(img, f"MAR: {mar_value:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # 顯示偵測狀態（中文用 PIL）
                status_text = "正常"
                status_color = (0, 255, 0)
                if model_alert:
                    status_text = "疲勞偵測!"
                    status_color = (255, 0, 0)
                elif ear_alert:
                    status_text = f"閉眼 {closed_duration:.1f}s"
                    status_color = (255, 165, 0)
                
                # 重新轉換為 PIL 並繪製中文
                img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(img_pil)
                draw.text((10, 120), f"狀態: {status_text}", font=font_small, fill=status_color)
                img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
                
                # 警報觸發
                if model_alert or ear_alert:
                    img = draw_alert(img, "請駕駛者休息，避免交通安全問題")

        # ------------------------------------------------
        
        # !!! 關鍵修改：必須回傳 av.VideoFrame !!!
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# -------------------------------------------------
# 4. Streamlit UI
# -------------------------------------------------
st.title("🚗 駕駛疲勞偵測系統 (WebRTC 版)")

# 側邊欄設定
with st.sidebar:
    st.header("參數設定")
    use_adaptive = st.checkbox("啟用自適應 EAR", value=True)
    if use_adaptive:
        calib_time = st.slider("校準時間(秒)", 3, 15, 5)
        st.info("💡 啟動攝影機後會自動開始校準")
    
    alert_th = st.slider("模型敏感度 (Prob)", 0.1, 0.9, 0.3)
    ear_th = st.slider("EAR 閾值", 0.15, 0.35, 0.25)
    closed_time = st.slider("閉眼警報時間(秒)", 0.5, 3.0, 1.0)

    mode = st.radio("模式", ["📷 即時偵測", "📂 影片上傳"])

# -------------------------------------------------
# 4-1. Webcam 模式
# -------------------------------------------------
if mode == "📷 即時偵測":
    # RTC 設定 (用於雲端部署時穿透防火牆)
    rtc_config = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

    ctx = webrtc_streamer(
        key="drowsiness-detection",
        video_processor_factory=DrowsinessProcessor,  # 指定上面的類別
        rtc_configuration=rtc_config,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    # 動態調整參數 (將 UI 數值傳入 Processor)
    if ctx.video_processor:
        ctx.video_processor.use_adaptive_ear = use_adaptive
        ctx.video_processor.alert_prob_threshold = alert_th
        ctx.video_processor.ear_threshold = ear_th
        ctx.video_processor.closed_seconds = closed_time
        if use_adaptive:
            ctx.video_processor.calibration_seconds = calib_time
        
        # 手動重新校準按鈕（如果需要）
        if use_adaptive:
            if st.button("🔄 重新校準"):
                ctx.video_processor.start_calibration()
                st.success("校準已重新啟動！")

# -------------------------------------------------
# 4-2. 影片上傳模式 (保持原樣，簡單處理)
# -------------------------------------------------
else:
    uploaded_file = st.file_uploader("上傳影片 (.mp4)", type=['mp4', 'avi'])
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)
        
        st_image = st.image([])
        
        # 這裡為了簡單，重複使用 init_mediapipe
        mp_mesh = init_mediapipe()
        local_model = load_model()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            frame = cv2.resize(frame, (640, 480))
            features, _ = extract_features(frame, mp_mesh)
            
            if features and local_model:
                prob = local_model.predict_proba([features])[0]
                if prob[1] > alert_th:
                     frame = draw_alert(frame, "請駕駛者休息，避免交通安全問題")
            
            st_image.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        cap.release()