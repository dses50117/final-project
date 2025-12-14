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
st.set_page_config(page_title="Driver Fatigue Detection System", layout="wide", page_icon="🚗")

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
        st.error("❌ Model file not found (drowsiness_model.pkl)")
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
    """Draw alert text and icon on image"""
    h, w, _ = image.shape
    
    # Create transparent overlay
    overlay = Image.new('RGBA', (w, h), (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)

    # Draw red triangle
    triangle_size = 80
    top_point = (w // 2, h // 2 - triangle_size // 2)
    left_point = (w // 2 - triangle_size // 2, h // 2 + triangle_size // 2)
    right_point = (w // 2 + triangle_size // 2, h // 2 + triangle_size // 2)
    draw.polygon([top_point, left_point, right_point], fill=(255, 0, 0, 180))
    
    # Draw exclamation mark inside triangle
    try:
        font_path = "arial.ttf"
        font = ImageFont.truetype(font_path, 60)
        draw.text((w // 2 - 10, h // 2 - 20), "!", font=font, fill=(255, 255, 255, 255))
    except IOError:
        draw.text((w // 2 - 10, h // 2 - 20), "!", fill=(255, 255, 255, 255))

    # Draw warning text
    try:
        font = ImageFont.truetype("arial.ttf", 30)
    except IOError:
        font = ImageFont.load_default()
        
    text_bbox = draw.textbbox((0,0), text, font=font)
    text_w = text_bbox[2] - text_bbox[0]
    text_h = text_bbox[3] - text_bbox[1]
    
    text_pos = (w // 2 - text_w // 2, h // 2 + triangle_size // 2 + 20)
    
    # Add semi-transparent background
    draw.rectangle(
        (text_pos[0] - 10, text_pos[1] - 10, text_pos[0] + text_w + 10, text_pos[1] + text_h + 10),
        fill=(0, 0, 0, 128)
    )
    draw.text(text_pos, text, font=font, fill=(255, 255, 0, 255))

    # Merge overlay back to original image
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
# 3. WebRTC Processor
# -------------------------------------------------
class DrowsinessProcessor(VideoProcessorBase):
    def __init__(self):
        # Parameter initialization
        self.use_adaptive_ear = True
        self.calibration_seconds = 10
        self.k_factor = 3.0  # Increased for more sensitive adaptive threshold
        self.ear_threshold = 0.25  # Raised to detect closed eyes more easily
        self.closed_seconds = 0.8  # Reduced to trigger alert faster
        self.alert_prob_threshold = 0.3
        self.consec_frames = 1
        
        # Yawn detection parameters
        self.mar_threshold = 0.6  # MAR threshold for yawn detection
        self.yawn_consec_frames = 3  # Consecutive frames needed to confirm yawn

        # State variables
        self.drowsy_count = 0
        self.closed_start = None
        self.yawn_count = 0  # Counter for consecutive yawn frames
        self.total_yawns = 0  # Total yawns detected
        
        # Initialize MediaPipe and model (independent instance per connection)
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=False, min_detection_confidence=0.5, min_tracking_confidence=0.5
        )
        self.model = load_model()

        # Calibration related
        self.is_calibrating = False
        self.calibration_start_time = None
        self.ear_samples = []
        self.adaptive_ear_threshold = None
        self.needs_calibration_start = True  # Auto-start calibration

    def start_calibration(self):
        self.is_calibrating = True
        self.calibration_start_time = time.time()
        self.ear_samples = []
        self.adaptive_ear_threshold = None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        # 1. Convert av.VideoFrame to numpy array (BGR)
        img = frame.to_ndarray(format="bgr24")
        h, w, _ = img.shape
        
        # Mirror flip (optional)
        img = cv2.flip(img, 1)
        
        # Auto-start calibration
        if self.use_adaptive_ear and self.needs_calibration_start:
            self.start_calibration()
            self.needs_calibration_start = False

        # 2. Feature extraction
        features, _ = extract_features(img, self.face_mesh)

        # 3. Logic processing (calibration or detection)
        # ------------------------------------------------
        if self.use_adaptive_ear and self.is_calibrating:
            # --- Calibration mode ---
            now = time.time()
            remain = self.calibration_seconds - (now - self.calibration_start_time)

            if remain > 0:
                if features:
                    ear_avg = (features[0] + features[1]) / 2.0
                    self.ear_samples.append(ear_avg)
                
                # Draw countdown text (using OpenCV for ASCII only)
                cv2.putText(img, f"Calibrating... {int(remain)}s", (30, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
                cv2.putText(img, "Keep eyes open naturally and blink normally", (30, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                # Calibration finished
                self.is_calibrating = False
                if self.ear_samples:
                    mean_ear = np.mean(self.ear_samples)
                    std_ear = np.std(self.ear_samples)
                    self.adaptive_ear_threshold = mean_ear - self.k_factor * std_ear
                    print(f"Calibration complete: Threshold={self.adaptive_ear_threshold}")

        else:
            # --- Detection mode ---
            current_th = self.ear_threshold
            if self.use_adaptive_ear and self.adaptive_ear_threshold is not None:
                current_th = self.adaptive_ear_threshold

            model_alert = False
            ear_alert = False
            yawn_alert = False
            
            if self.model is not None and features:
                ear_avg = (features[0] + features[1]) / 2.0
                mar_value = features[2]
                now = time.time()

                # XGBoost prediction
                # features: [L_EAR, R_EAR, MAR, P, Y, R]
                try:
                    input_feat = np.array([features])
                    pred = self.model.predict(input_feat)[0]
                    prob = self.model.predict_proba(input_feat)[0]
                    
                    # Model decision logic
                    if pred == 1 and prob[1] >= self.alert_prob_threshold:
                        self.drowsy_count += 1
                    else:
                        self.drowsy_count = 0
                    
                    if self.drowsy_count >= self.consec_frames:
                        model_alert = True
                except Exception as e:
                    print(f"Prediction Error: {e}")
                    prob = [0.0, 0.0]  # Default value on error

                # EAR-based logic (eyes closed too long)
                if ear_avg < current_th:
                    if self.closed_start is None:
                        self.closed_start = now
                else:
                    self.closed_start = None

                closed_duration = (now - self.closed_start) if self.closed_start else 0
                if closed_duration >= self.closed_seconds:
                    ear_alert = True

                # Yawn detection logic (MAR-based)
                if mar_value > self.mar_threshold:
                    self.yawn_count += 1
                    if self.yawn_count >= self.yawn_consec_frames:
                        yawn_alert = True
                        if self.yawn_count == self.yawn_consec_frames:  # Only count once per yawn
                            self.total_yawns += 1
                else:
                    self.yawn_count = 0
                
                # Draw metrics (using OpenCV for English text)
                cv2.putText(img, f"Prob: {prob[1]:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(img, f"EAR: {ear_avg:.2f} (Th:{current_th:.2f})", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(img, f"MAR: {mar_value:.2f} (Th:{self.mar_threshold:.2f})", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(img, f"Yawns: {self.total_yawns}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Display detection status
                status_text = "Normal"
                status_color = (0, 255, 0)
                if model_alert:
                    status_text = "Fatigue Detected!"
                    status_color = (0, 0, 255)  # Red in BGR
                elif ear_alert:
                    status_text = f"Eyes Closed {closed_duration:.1f}s"
                    status_color = (0, 165, 255)  # Orange in BGR
                elif yawn_alert:
                    status_text = "Yawning Detected!"
                    status_color = (0, 255, 255)  # Yellow in BGR
                
                cv2.putText(img, f"Status: {status_text}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
                
                # Trigger alert
                if model_alert or ear_alert:
                    img = draw_alert(img, "WARNING: Driver Fatigue Detected!")
                elif yawn_alert:
                    img = draw_alert(img, "YAWN DETECTED: Take a break!")

        # ------------------------------------------------
        
        # Return av.VideoFrame
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# -------------------------------------------------
# 4. Streamlit UI
# -------------------------------------------------
st.title("🚗 Driver Fatigue Detection System")

# Sidebar settings
with st.sidebar:
    st.header("Settings")
    use_adaptive = st.checkbox("Enable Adaptive EAR", value=True)
    if use_adaptive:
        calib_time = st.slider("Calibration Time (seconds)", 3, 15, 5)
        st.info("💡 Calibration starts automatically when camera is enabled")
    
    alert_th = st.slider("Model Sensitivity (Prob)", 0.1, 0.9, 0.5)
    ear_th = st.slider("EAR Threshold", 0.15, 0.50, 0.25)  # Increased default and max
    closed_time = st.slider("Eyes Closed Alert Time (seconds)", 0.3, 3.0, 0.8)  # Reduced default
    mar_th = st.slider("MAR Yawn Threshold", 0.5, 0.8, 0.65)
    yawn_frames = st.slider("Yawn Consecutive Frames", 1, 5, 4)

    mode = st.radio("Mode", ["📷 Live Detection", "📂 Video Upload"])

# -------------------------------------------------
# 4-1. Webcam Mode
# -------------------------------------------------
if mode == "📷 Live Detection":
    # RTC configuration (for cloud deployment firewall penetration)
    rtc_config = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

    ctx = webrtc_streamer(
        key="drowsiness-detection",
        video_processor_factory=DrowsinessProcessor,  # Specify the class above
        rtc_configuration=rtc_config,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    # Dynamically adjust parameters (pass UI values to Processor)
    if ctx.video_processor:
        ctx.video_processor.use_adaptive_ear = use_adaptive
        ctx.video_processor.alert_prob_threshold = alert_th
        ctx.video_processor.ear_threshold = ear_th
        ctx.video_processor.closed_seconds = closed_time
        ctx.video_processor.mar_threshold = mar_th
        ctx.video_processor.yawn_consec_frames = yawn_frames
        if use_adaptive:
            ctx.video_processor.calibration_seconds = calib_time
        
        # Manual recalibration button (if needed)
        if use_adaptive:
            if st.button("🔄 Recalibrate"):
                ctx.video_processor.start_calibration()
                st.success("Calibration restarted!")

# -------------------------------------------------
# 4-2. Video Upload Mode
# -------------------------------------------------
else:
    st.subheader("📂 Video Upload Mode")
    uploaded_file = st.file_uploader("Upload Video (.mp4, .avi)", type=['mp4', 'avi'])
    
    if uploaded_file:
        # Save uploaded file temporarily
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        tfile.close()
        
        # Create placeholder for video display (will show processed frames here)
        video_placeholder = st.empty()
        
        # Show first frame as preview
        cap_preview = cv2.VideoCapture(tfile.name)
        ret, first_frame = cap_preview.read()
        if ret:
            first_frame = cv2.resize(first_frame, (640, 480))
            video_placeholder.image(cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)
        cap_preview.release()
        
        # Processing controls
        col1, col2 = st.columns(2)
        with col1:
            process_btn = st.button("▶️ Start Processing")
        with col2:
            download_processed = st.checkbox("Save processed video", value=True)
        
        if process_btn:
            # Initialize MediaPipe and model
            mp_mesh = init_mediapipe()
            local_model = load_model()
            
            # Open video capture
            cap = cv2.VideoCapture(tfile.name)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Always save processed video
            output_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            # Use mp4v codec (most compatible)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_file.name, fourcc, fps, (640, 480))
            
            # UI elements for processing
            st.write(f"**Video Info**: {total_frames} frames @ {fps:.2f} FPS")
            st.info("⚡ Processing video quickly - no live preview for maximum speed!")
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Yawn detection state
            yawn_count = 0
            total_yawns = 0
            
            # Process video frame by frame
            frame_num = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_num += 1
                frame = cv2.resize(frame, (640, 480))
                
                # Extract features and detect
                features, _ = extract_features(frame, mp_mesh)
                
                if features and local_model:
                    ear_avg = (features[0] + features[1]) / 2.0
                    mar_value = features[2]
                    
                    # Model prediction
                    try:
                        prob = local_model.predict_proba([features])[0]
                        
                        # Display metrics
                        cv2.putText(frame, f"Prob: {prob[1]:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(frame, f"EAR: {ear_avg:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(frame, f"MAR: {mar_value:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        # Yawn detection
                        if mar_value > mar_th:
                            yawn_count += 1
                            if yawn_count >= yawn_frames:
                                if yawn_count == yawn_frames:  # Count once per yawn
                                    total_yawns += 1
                                frame = draw_alert(frame, "YAWN DETECTED: Take a break!")
                                cv2.putText(frame, f"Status: Yawning!", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        else:
                            yawn_count = 0
                            # Fatigue detection
                            if prob[1] > alert_th:
                                frame = draw_alert(frame, "WARNING: Driver Fatigue Detected!")
                                cv2.putText(frame, f"Status: Fatigue Detected!", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            else:
                                cv2.putText(frame, f"Status: Normal", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        cv2.putText(frame, f"Total Yawns: {total_yawns}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                    except Exception as e:
                        cv2.putText(frame, f"Error: {str(e)[:30]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                
                # Save processed frame
                video_writer.write(frame)
                
                # Update progress less frequently for speed
                if frame_num % 50 == 0 or frame_num == total_frames:
                    progress = frame_num / total_frames
                    progress_bar.progress(progress)
                    status_text.text(f"Processing: Frame {frame_num}/{total_frames} | Yawns detected: {total_yawns}")
            
            # Cleanup
            cap.release()
            video_writer.release()
            
            # Show completion message
            st.success(f"✅ Processing complete! Total yawns detected: {total_yawns}")
            
            # Display processed video
            st.subheader("📹 Processed Video with Detection")
            st.info("💡 If video doesn't play in browser, please download it to view")
            video_placeholder.empty()  # Clear the preview
            
            # Try to display video
            try:
                st.video(output_file.name)
            except Exception as e:
                st.warning(f"Could not display video in browser: {e}")
            
            # Provide download button
            if download_processed:
                with open(output_file.name, 'rb') as f:
                    st.download_button(
                        label="⬇️ Download Processed Video",
                        data=f,
                        file_name="processed_video.mp4",
                        mime="video/mp4"
                    )
        
        # Cleanup temp file
        try:
            os.unlink(tfile.name)
        except:
            pass