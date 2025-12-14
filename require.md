1. CRISP-DM 專案分析 (針對 NTHU + DDD 資料集)
1. Business Understanding (商業理解)

目標：整合多來源資料，建立一個高魯棒性的疲勞偵測模型，並在 Streamlit 上實現「即時 Webcam」與「影片檔」雙重監控。


策略：利用 PPT 提及的 "MediaPipe + XGBoost" 輕量化策略，確保在沒有 GPU 的筆電上也能跑出 30 FPS 。


2. Data Understanding (資料理解)

資料來源 A (NTHU-DDD2)：模擬駕駛室環境，包含夜間、戴眼鏡等困難樣本 (約 6.6 萬張)。

資料來源 B (DDD)：背景較單純，特徵清晰 (約 4.1 萬張)。

整合挑戰：兩者解析度不同，但因為我們是用 MediaPipe 抓「相對座標 (Landmarks)」，所以解析度差異不影響，可以直接合併訓練。

3. Data Preparation (資料準備)

特徵工程：不使用原始像素 (Pixels)，而是提取 EAR (眼睛縱橫比) 與 MAR (嘴巴縱橫比)。

清洗：需剔除 MediaPipe 抓不到臉的圖片 (Outliers)。

4. Modeling (模型建立)

演算法：XGBoost Classifier。它能很好地處理這 10 萬筆結構化數據 (Table data)，訓練速度快且準確。

5. Evaluation (評估)

使用混淆矩陣 (Confusion Matrix) 確認模型是否能正確區分 NTHU 的困難樣本。

6. Deployment (部署)

Streamlit App：設計「即時監控模式」與「影片分析模式」。

2. VS Code 專案實作 (完整程式碼)
請在 VS Code 建立一個資料夾，並確認已安裝套件： pip install opencv-python mediapipe pandas scikit-learn xgboost streamlit joblib tqdm

步驟一：資料整理與特徵提取 (1_process_data.py)
這個腳本會自動讀取兩個資料集，提取特徵並合併成一個 CSV。

⚠️ 請注意資料夾結構設定： 假設您的目錄結構如下 (請依照您的實際路徑修改 DATASETS 變數)：

Plaintext

Project/
├── raw_data/
│   ├── nthuddd2/
│   │   ├── Drowsy/ (放圖片)
│   │   └── Non Drowsy/
│   ├── ddd/
│   │   ├── Drowsy/
│   │   └── Non Drowsy/
Python

# 1_process_data.py
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
from scipy.spatial import distance as dist
from tqdm import tqdm  # 進度條

# --- 設定路徑 (請依您的實際下載位置修改) ---
DATASETS = {
    "nthu": {
        "drowsy": "raw_data/nthuddd2/Drowsy",
        "alert": "raw_data/nthuddd2/Non Drowsy"
    },
    "ddd": {
        "drowsy": "raw_data/ddd/Drowsy",
        "alert": "raw_data/ddd/Non Drowsy"
    }
}

# --- MediaPipe 設定 ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,  # 處理靜態圖片模式
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

# --- 關鍵點索引 ---
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH = [78, 81, 13, 311, 308, 402, 14, 178]

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

data_list = []

print("🚀 開始處理兩個資料集，這可能需要幾分鐘...")

# 遍歷所有資料集與類別
for dataset_name, paths in DATASETS.items():
    for label_name, folder_path in paths.items():
        label = 1 if label_name == "drowsy" else 0
        
        if not os.path.exists(folder_path):
            print(f"⚠️ 跳過：找不到路徑 {folder_path}")
            continue

        files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"正在處理 {dataset_name} - {label_name} (共 {len(files)} 張)...")

        for filename in tqdm(files):
            filepath = os.path.join(folder_path, filename)
            image = cv2.imread(filepath)
            if image is None: continue

            results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                
                ear_left = calculate_ear(LEFT_EYE, landmarks)
                ear_right = calculate_ear(RIGHT_EYE, landmarks)
                mar = calculate_mar(MOUTH, landmarks)
                
                # 加入資料列表
                data_list.append([ear_left, ear_right, mar, label])

# 轉為 DataFrame 並儲存
df = pd.DataFrame(data_list, columns=['ear_left', 'ear_right', 'mar', 'label'])
df.to_csv('combined_dataset.csv', index=False)
print(f"✅ 處理完成！共提取 {len(df)} 筆有效資料，已存為 combined_dataset.csv")
步驟二：訓練模型 (2_train_model.py)
因為資料量大 (10萬筆)，我們使用 XGBoost。

Python

# 2_train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# 1. 讀取整合後的資料
print("📂 讀取資料集...")
try:
    df = pd.read_csv('combined_dataset.csv')
except FileNotFoundError:
    print("❌ 找不到 combined_dataset.csv，請先執行步驟一！")
    exit()

print(f"總資料量: {len(df)}")
print("類別分佈:\n", df['label'].value_counts())

# 2. 切分資料
X = df[['ear_left', 'ear_right', 'mar']]
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 訓練 (使用輕量化參數以避免過擬合)
print("🧠 開始訓練 XGBoost...")
model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6,
    use_label_encoder=False,
    eval_metric='logloss'
)
model.fit(X_train, y_train)

# 4. 評估
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n🏆 模型準確率: {acc:.4f}")
print("\n詳細報告:\n", classification_report(y_test, y_pred))
print("\n混淆矩陣:\n", confusion_matrix(y_test, y_pred))

# 5. 儲存
joblib.dump(model, 'driver_drowsiness_model.pkl')
print("💾 模型已儲存為 driver_drowsiness_model.pkl")
步驟三：Streamlit 終極 Demo (app.py)
這個 App 包含：

側邊欄切換：Webcam 監控 / 影片檔分析。

即時警示機制：不只是單幀判斷，我加入了一個 counter 機制，連續偵測到疲勞才報警 (模擬 PERCLOS 概念)，避免誤報。

影片支援：可以上傳 .mp4 進行分析。

Python

# app.py
import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import joblib
import tempfile
import time
from scipy.spatial import distance as dist
from PIL import Image

# --- 配置與載入 ---
st.set_page_config(page_title="疲勞駕駛偵測系統 Pro", layout="wide", page_icon="🚗")

@st.cache_resource
def load_model():
    return joblib.load('driver_drowsiness_model.pkl')

try:
    model = load_model()
except:
    st.error("⚠️ 找不到模型檔，請先執行訓練程式！")
    st.stop()

# MediaPipe 設定
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 關鍵點定義
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH = [78, 81, 13, 311, 308, 402, 14, 178]

# --- 輔助函式 ---
def calculate_features(landmarks):
    # EAR
    def eye_aspect_ratio(eye_pts):
        p = [landmarks[i] for i in eye_pts]
        A = dist.euclidean((p[1].x, p[1].y), (p[5].x, p[5].y))
        B = dist.euclidean((p[2].x, p[2].y), (p[4].x, p[4].y))
        C = dist.euclidean((p[0].x, p[0].y), (p[3].x, p[3].y))
        return (A + B) / (2.0 * C) if C != 0 else 0

    # MAR
    def mouth_aspect_ratio(mouth_pts):
        p = [landmarks[i] for i in mouth_pts]
        A = dist.euclidean((p[1].x, p[1].y), (p[7].x, p[7].y))
        B = dist.euclidean((p[2].x, p[2].y), (p[6].x, p[6].y))
        C = dist.euclidean((p[3].x, p[3].y), (p[5].x, p[5].y))
        D = dist.euclidean((p[0].x, p[0].y), (p[4].x, p[4].y))
        return (A + B + C) / (2.0 * D) if D != 0 else 0

    ear_left = eye_aspect_ratio(LEFT_EYE)
    ear_right = eye_aspect_ratio(RIGHT_EYE)
    mar = mouth_aspect_ratio(MOUTH)
    return ear_left, ear_right, mar

def draw_landmarks(image, landmarks):
    h, w, _ = image.shape
    for idx in LEFT_EYE + RIGHT_EYE + MOUTH:
        pt = landmarks[idx]
        cv2.circle(image, (int(pt.x * w), int(pt.y * h)), 1, (0, 255, 255), -1)

# --- 介面設計 ---
st.title("🚗 AI 駕駛疲勞監控系統 (NTHU+DDD)")
st.sidebar.title("控制面板")
mode = st.sidebar.radio("選擇模式", ["📷 即時 Webcam 監控", "📂 影片檔案分析"])

# 狀態變數 (用於平滑化預測，避免閃爍)
if 'drowsy_counter' not in st.session_state:
    st.session_state.drowsy_counter = 0

ALARM_TRIGGER_FRAMES = 5  # 連續 N 幀偵測到疲勞才報警

# --- 主邏輯: 處理單幀影像 ---
def process_frame(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    
    status = "檢測中..."
    color = (255, 255, 0)
    features_info = {"EAR": 0.0, "MAR": 0.0}
    is_drowsy = False

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        
        # 1. 提取特徵
        ear_l, ear_r, mar = calculate_features(landmarks)
        features_info = {"EAR": (ear_l + ear_r)/2, "MAR": mar}
        
        # 2. 視覺化關鍵點
        draw_landmarks(frame, landmarks)
        
        # 3. 模型預測
        input_data = np.array([[ear_l, ear_r, mar]])
        prediction = model.predict(input_data)[0]
        
        # 4. 警報邏輯 (平滑化)
        if prediction == 1:
            st.session_state.drowsy_counter += 1
        else:
            st.session_state.drowsy_counter = max(0, st.session_state.drowsy_counter - 1)

        if st.session_state.drowsy_counter >= ALARM_TRIGGER_FRAMES:
            status = "⚠️ 疲勞駕駛警告! (DROWSY)"
            color = (0, 0, 255) # 紅色
            is_drowsy = True
        else:
            status = "✅ 精神狀態良好 (ALERT)"
            color = (0, 255, 0) # 綠色

    # 繪製文字
    cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(frame, f"EAR: {features_info['EAR']:.2f} | MAR: {features_info['MAR']:.2f}", 
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    return frame, is_drowsy, features_info

# --- 模式 A: Webcam ---
if mode == "📷 即時 Webcam 監控":
    col1, col2 = st.columns([3, 1])
    with col1:
        st_frame = st.image([])
    with col2:
        st.markdown("### 狀態數據")
        kpi_status = st.empty()
        kpi_ear = st.metric("平均 EAR (眼)", "0.00")
        kpi_mar = st.metric("MAR (嘴)", "0.00")
    
    run = st.checkbox("啟動鏡頭", value=False)
    cap = cv2.VideoCapture(0)
    
    while run:
        ret, frame = cap.read()
        if not ret:
            st.error("無法讀取鏡頭")
            break
        
        frame = cv2.flip(frame, 1)
        processed_frame, is_drowsy, info = process_frame(frame)
        
        st_frame.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB))
        
        if is_drowsy:
            kpi_status.error("⚠️ 警告！")
        else:
            kpi_status.success("正常")

    cap.release()

# --- 模式 B: 影片分析 ---
elif mode == "📂 影片檔案分析":
    uploaded_file = st.sidebar.file_uploader("上傳影片檔 (.mp4)", type=["mp4", "avi"])
    
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        
        cap = cv2.VideoCapture(tfile.name)
        st_frame = st.image([])
        progress_bar = st.progress(0)
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        curr_frame = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            processed_frame, is_drowsy, info = process_frame(frame)
            st_frame.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB))
            
            curr_frame += 1
            progress_bar.progress(min(curr_frame / total_frames, 1.0))
            
        cap.release()