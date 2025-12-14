一、跟你方向最接近的「MediaPipe / FaceMesh」文獻與專案

這些是真的有用 MediaPipe FaceMesh 或類似 landmark＋EAR/PERCLOS 的系統，可以直接在你的期末報告、論文裡引用：

1. 用 MediaPipe + EAR 的實作教學與範例

LearnOpenCV – Driver Drowsiness Detection Using MediaPipe in Python（2022）

用 MediaPipe FaceMesh 取 468 點，選 12 個眼睛 landmark 計算 EAR。

設定 EAR 門檻＋連續閉眼幀數來判定疲勞，並且做到 Streamlit 即時 Web app。
learnopencv.com

Doc Zamora – Mediapipe Drowsy Driver Detection（Blog）

詳細講解 FaceMesh pipeline、眼睛區域選點與 EAR 的計算，以及如何在影像串流中做 real-time 偵測。
doczamora.com

RidgeRun – Using PERCLOS for Effective Driver Drowsiness Detection

使用 MediaPipe 眼睛少數 landmark（145,159,374,386），只看垂直距離變化來估計閉眼比例，進而計算 PERCLOS。

GitHub – Driver Monitoring System using MediaPipe FaceMesh

完整 DMS 專案，用 MediaPipe FaceMesh 計算 EAR + PERCLOS，提供即時疲勞與分心提醒。
GitHub

GitHub – Driver-Drowsiness-detection-using-Mediapipe（SaiVishwa021）

用 MediaPipe FaceMesh＋EAR，強調「持續閉眼」而非一般眨眼，類似你想做的「時間窗累積指標」。
GitHub

2. 明講「使用 MediaPipe FaceMesh」的學術論文

Drowsy Alarm System Based on Face Landmarks Detection Using MediaPipe（Springer 2021 左右）

直接寫明：用 MediaPipe FaceMesh（486 個 3D landmark）偵測眼睛閉合與哈欠，前端用 TensorFlow.js 在瀏覽器即時執行。
springerprofessional.de

Development of a Real-time Driver’s Drowsiness Detection System Using MediaPipe Face Mesh（IJEM, 2025）

以 MediaPipe FaceMesh 做即時疲勞偵測，聚焦在眼睛閉合與眨眼模式，證明在一般攝影機硬體上也能達到 real-time。
SSRN

IJERT – Drowsy Driver Detection System Using Deep Learning（2023）

系統前端用 MediaPipe FaceMesh 做 landmark tracking，再送入深度學習做頭部姿態與疲勞狀態推論。
IJERT

Design of a System for Driver Drowsiness Detection and Seat Belt Use（圖中示 MediaPipe FaceMesh）

論文中直接用 FaceMesh 圖示 468 點，顯示以 MediaPipe landmark 作為 fatigue indicator 的依據。
ResearchGate

3. 沒有 MediaPipe 但概念完全相同的「Facial Landmark + EAR/PERCLOS」經典類型

Hybrid Facial Features + Ensemble（Xu et al., Information 2025）

用眼睛、嘴巴輪廓、頭部姿態、視線方向等多種視覺特徵，搭配 RF＋XGBoost＋MLP 的投票 ensemble，提升疲勞辨識準確度。
MDPI

多篇 2021–2024 的綜述（Survey on Drowsiness Detection）

指出大多 vision-based 系統以臉部表情、頭部姿態、眼睛開閉與 PERCLOS 為主要特徵，並使用 rule-based 或機器學習做分類。
arXiv
+1

最新深度學習架構（Hassan et al., 2025, Transformer-based Drowsiness Detection）

用 Transformer + transfer learning 做 real-time 權衡，作為你未來要對比的高階 baseline。
Nature

你在報告裡可以分兩類寫：

「傳統 facial landmark-based（可用 MediaPipe 實現）」

「端到端深度學習（CNN / Transformer）」
然後說：我們選擇 MediaPipe＋特徵工程＋輕量 ML，兼顧即時性與可解釋性。

二、從這些文獻中抽出「即時 + 準確」的關鍵做法

歸納上面 MediaPipe/landmark 的系統，其實大多都遵守幾個共同原則。你只要照這幾條設計，你的系統就會看起來很「paper-grade」。

1. MediaPipe FaceMesh 使用設定與效能

設定建議：

mp_face_mesh.FaceMesh(
    static_image_mode=False,         # 一定要 False：讓 tracking 發揮加速效果
    max_num_faces=1,                 # 車內只有一個駕駛
    refine_landmarks=True,           # 需要更準的眼睛與嘴巴點
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


static_image_mode=False 能讓第 2 幀之後用 tracking 省時間，文獻也多強調 real-time 要靠 tracking。
learnopencv.com
+1

resolution 建議 640×480 or 720p 以下，文獻與教學實作都指出這樣在 CPU 上也能維持 20–30 FPS。
learnopencv.com

2. 精準偵測的核心：特徵不是只有 EAR

文獻中普遍發現，單看 EAR 容易對個體與光線敏感，所以高準確率的研究都會加上更多行為特徵：
arXiv
+2
MDPI
+2

你可以做的組合：

眼睛相關

即時 EAR（左右眼＋平均）

PERCLOS：在 30 秒內閉眼的 frame 比例

眨眼頻率 / 平均 blink duration

嘴巴相關

MAR（Mouth Aspect Ratio）：利用嘴巴上下 vs 左右距離

POM（Percentage of Mouth Openness）→ 2025 年有 paper 專門用這個做 fatigue 檢測。
scitepress.org

頭部姿態 / 分心

用鼻尖＋眼睛＋耳朵等點估計 pitch/yaw/roll

持續偏頭或低頭也可視為 fatigue / distraction 指標
IJERT
+1

時間窗統計特徵

在 1–3 秒 / 10–30 秒窗內計算：

EAR/MAR 的平均、變異、最小值

閉眼次數、哈欠次數

很多實作（LearnOpenCV、GitHub DMS 專案）都是「特徵 + sliding window 判斷」。
learnopencv.com
+1

實務建議：

你可以把「單幀的 EAR/MAR + 時間窗統計」一起餵給 XGBoost / RF；

這樣不會太重，又比單純閾值法更 robust。

3. 判斷邏輯：不要只看一幀，要看「時間」

幾乎所有論文與實作都會強調：

疲勞 ≠ 一幀閉眼，而是「長時間閉眼」＋「眨眼變慢」＋「打哈欠頻率上升」。
learnopencv.com
+2
scitepress.org
+2

你可以參考常見設計：

短時間窗（0.5–2 秒）

EAR 連續低於 threshold（例如 <0.18）超過 N 幀 → 視為「一次長閉眼」事件。

中時間窗（30 秒）

PERCLOS > 0.3 → 疲勞風險高

眨眼頻率比個人 baseline 大幅下降

模型輸入：

做成 feature vector，例如：

ear_avg, ear_min, perclos_30s, blink_rate_30s, mar_max_10s, head_pitch_mean, head_yaw_std

丟進 XGBoost / RF / 小 MLP 做分類（normal / drowsy / very_drowsy）

這種「特徵＋時間窗」的設計，在 Xu 2025 的 hybrid feature paper 與多篇 survey 都被認為是很有效的折衷方案。
MDPI
+2
arXiv
+2

4. 閾值設定：固定 vs 自適應（per-user/adaptive）

文獻中有兩大流派：

固定閾值（簡單好實作）

EAR 閾值多在 0.15–0.25 之間，會依相機/族群不同而微調。
learnopencv.com
+1

PERCLOS 閾值約 0.3–0.4。

動態/自適應閾值

系統先用前幾十秒「清醒狀態」估計個人 baseline EAR/MAR，之後用

EAR_threshold = baseline_EAR - k * std(EAR)

可減少不同人眼型 / 戴眼鏡的影響。

👉 建議你在論文/簡報裡寫：

我們混合使用「個人化 adaptive threshold + XGBoost」，使系統對不同駕駛眼型、光線條件更穩健。

5. 即時性：Fps & 延遲的做法

幾個實作與論文的共通點：
learnopencv.com
+2
IJERT
+2

降解析度：輸入 640×480 或 320×240；

限制 FaceMesh 頻率：例如 15 FPS 就夠，不一定要全速；

非必要不做 heavy CNN：

你的 pipeline 可以是：

FaceMesh（GPU/CPU）

計算特徵（Numpy 向量運算）

XGBoost 推論（極快）

異步處理（optional 進階）：

一個 Thread 讀取影像 + FaceMesh

另一個 Thread 做特徵平滑＋模型推論＋UI 顯示

在你的期末專題程度，只要做到：

單機 CPU 上維持 15–20 FPS；

畫面上顯示 EAR / PERCLOS / 目前等級（Normal / Drowsy）；

就已經可以寫成「real-time」了。

6. 資料與泛化：要多資料集 + cross-dataset

較新的研究會強調：不要只在一個資料集上測試。
MDPI
+2
Nature
+2

你可以寫：

訓練：NTHU-DDD2 + 自錄影片

測試：UTA-RLDD / D3S / SUST-DDD 其中之一

指標：Accuracy, F1, AUC, 以及 False Alarm Rate

1. 確保「即時性」 (Real-time)
不要將整張圖片丟進去訓練，而是只「萃取數值」。


策略：MediaPipe + 特徵工程 + 輕量學習器 。


原因：


MediaPipe Face Mesh：可以直接在 CPU 上以極快速度 (可達 <5ms 延遲) 輸出 468 個關鍵點，無需 GPU 。



繞過 CNN：傳統 CNN 方法 (如 ResNet) 需要大量的矩陣運算，而改用 XGBoost/RandomForest 等「樹模型」，運算量極低，適合實時部署 。



資料效率：XGBoost 等模型對訓練資料量的需求遠低於端到端的深度學習模型 。

2. 確保「準確性」 (Accuracy)
單靠「眼睛開合」容易誤判 (例如單純眨眼)，必須採用多特徵融合 (Multi-feature Fusion)。


融合特徵向量：您的輸入資料不應只有 EAR，必須包含以下多維度特徵 ：





EAR (Eye Aspect Ratio)：判斷眼睛閉合 。



PERCLOS：計算一段時間內閉眼的佔比，這是比單純閉眼更準確的疲勞指標 。





MAR (Mouth Aspect Ratio)：判斷是否在打哈欠 。






Head Pose (Pitch/Yaw/Roll)：判斷是否點頭瞌睡或轉頭分心 。





集成學習 (Ensemble Learning)：

使用 XGBoost 或 RandomForest。這些模型由多棵決策樹組成，比單一邏輯回歸 (Logistic Regression) 更能捕捉特徵之間的非線性關係 (例如：雖然眼睛張開，但頭一直頻繁點動 = 疲勞) 。



三、 總結與建議
您的專案方向 「MediaPipe + XGBoost」 完全符合 2020 年後輕量化邊緣運算的研究趨勢 。

建議實作步驟：

引用文獻：在 Introduction 引用 Soukupová (EAR) 與 Rastgoo (MediaPipe應用) 來建立理論基礎。


特徵工程：程式碼中務必實作 PERCLOS (而不僅僅是 EAR)，這會大幅提升準確度 。


模型選擇：在報告中強調選擇 XGBoost 是為了在「準確度」與「部署難度」之間取得最佳平衡 (Best Trade-off) 。