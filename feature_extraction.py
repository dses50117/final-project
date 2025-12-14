import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
from scipy.spatial import distance as dist
from tqdm import tqdm  # 進度條
from pathlib import Path
from mediapipe.python._framework_bindings import resource_util

# =======================================================
# 0. MediaPipe 資源路徑處理（你原本的設定保留）
# =======================================================
DEFAULT_RESOURCE_DIR = Path(mp.__file__).resolve().parent.parent  # site-packages
FALLBACK_RESOURCE_DIR = Path("C:/mp_resources")
RESOURCE_DIR = FALLBACK_RESOURCE_DIR if (FALLBACK_RESOURCE_DIR / "mediapipe").exists() else DEFAULT_RESOURCE_DIR
_set_dir = resource_util.set_resource_dir
resource_util.set_resource_dir = lambda *_args, **_kwargs: _set_dir(str(RESOURCE_DIR))  # override to force ASCII-safe path
resource_util.set_resource_dir(str(RESOURCE_DIR))

# =======================================================
# 1. 路徑與「每類最多處理張數」設定
# =======================================================
DATASETS = {
    "drowsy": "data/drowsy",
    "notdrowsy": "data/notdrowsy"
}

# ★ 每一類最多處理幾張圖（可以依需求調整）
MAX_PER_CLASS = 5000   # 想要全跑就改成 None 或很大的數字


# =======================================================
# 2. MediaPipe FaceMesh 設定（改成比較快的版本）
# =======================================================
mp_face_mesh = mp.solutions.face_mesh

graph_override = FALLBACK_RESOURCE_DIR / "mediapipe/modules/face_landmark/face_landmark_front_cpu.binarypb"
if graph_override.exists():
    mp_face_mesh._BINARYPB_FILE_PATH = str(graph_override)

# ★ 關掉 refine_landmarks，加速推論
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,          # 處理靜態圖片
    max_num_faces=1,
    refine_landmarks=False,          # ★ 關掉高精度細節 → 速度大幅提升
    min_detection_confidence=0.5
)

# =======================================================
# 3. 關鍵點索引 & EAR / MAR 計算
# =======================================================
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH = [78, 81, 13, 311, 308, 402, 14, 178]
POSE_POINTS = [33, 263, 1, 61, 291, 199]

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

# =======================================================
# 4. 主迴圈：逐類別處理圖片
# =======================================================
for label_name, folder_path in DATASETS.items():
    label = 1 if label_name == "drowsy" else 0

    if not os.path.exists(folder_path):
        print(f"⚠️ 跳過：找不到路徑 {folder_path}")
        continue

    files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # ★ 如果有設定 MAX_PER_CLASS，就只取前 N 張
    if MAX_PER_CLASS is not None:
        files = files[:MAX_PER_CLASS]

    print(f"正在處理 {label_name} (這次實際處理 {len(files)} 張)...")

    # tqdm 顯示進度條
    for filename in tqdm(files):
        filepath = os.path.join(folder_path, filename)
        image = cv2.imread(filepath)
        if image is None:
            continue

        # ★ 可選：若圖片太大，先縮小以加速（會更快）
        h, w = image.shape[:2]
        if max(h, w) > 720:  # 例如最大邊限制在 720
            scale = 720 / max(h, w)
            new_size = (int(w * scale), int(h * scale))
            image = cv2.resize(image, new_size)

        # FaceMesh 要吃 RGB
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

            ear_left = calculate_ear(LEFT_EYE, landmarks)
            ear_right = calculate_ear(RIGHT_EYE, landmarks)
            mar = calculate_mar(MOUTH, landmarks)

            # Head pose (pitch, yaw, roll)
            face_2d, face_3d = [], []
            for idx in POSE_POINTS:
                lm = landmarks[idx]
                x, y = int(lm.x * w), int(lm.y * h)
                face_2d.append([x, y])
                face_3d.append([x, y, lm.z])

            face_2d = np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)

            cam_matrix = np.array([[w, 0, h / 2],
                                   [0, w, w / 2],
                                   [0, 0, 1]], dtype=np.float64)
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            pitch = yaw = roll = 0.0
            try:
                success, rot_vec, _ = cv2.solvePnP(
                    face_3d, face_2d, cam_matrix, dist_matrix, flags=cv2.SOLVEPNP_ITERATIVE
                )
                if success:
                    rmat, _ = cv2.Rodrigues(rot_vec)
                    angles, *_ = cv2.RQDecomp3x3(rmat)
                    pitch = angles[0] * 360
                    yaw = angles[1] * 360
                    roll = angles[2] * 360
            except Exception:
                pitch = yaw = roll = 0.0

            data_list.append([ear_left, ear_right, mar, pitch, yaw, roll, label])

# =======================================================
# 5. 存成 CSV
# =======================================================
df = pd.DataFrame(data_list, columns=['ear_left', 'ear_right', 'mar', 'pitch', 'yaw', 'roll', 'label'])
df.to_csv('training_data.csv', index=False, encoding='utf-8-sig')
print(f"✅ 處理完成！共提取 {len(df)} 筆有效資料，已存為 training_data.csv")
