# 駕駛疲勞偵測系統 (Driver Drowsiness Detection System)

這是一個使用電腦視覺和機器學習技術，透過攝影機即時偵測駕駛員是否疲勞的系統。

## 功能

- **即時偵測**: 可透過電腦攝影機 (Webcam) 或上傳的影片檔案進行即時分析。
- **多特徵融合**:
  - **眼睛開合程度 (EAR - Eye Aspect Ratio)**: 分析眼睛的睜開程度，判斷是否閉眼或頻繁眨眼。
  - **打哈欠偵測 (MAR - Mouth Aspect Ratio)**: 分析嘴巴的張開程度，判斷是否在打哈欠。
  - **頭部姿態 (Head Pose)**: 分析頭部的俯仰、偏航和滾動角度，判斷是否點頭或歪頭。
- **機器學習模型**: 使用 XGBoost 分類器對上述特徵進行綜合判斷，提供更準確的疲勞預測。
- **雙重警示機制**:
  1.  **模型警示**: 當模型連續在數個影格中偵測到高疲勞機率時觸發。
  2.  **閉眼警示**: 當駕駛員連續閉眼超過設定的秒數時觸發。
- **視覺化介面**: 使用 Streamlit 建立的互動式網頁介面，可調整警示參數並即時顯示偵測結果與疲勞機率。

## 專案結構

```
project/
│
├── app.py                  # Streamlit 主應用程式
├── train_model.py          # 模型訓練腳本
├── feature_extraction.py   # 從圖片資料集提取特徵的腳本
│
├── drowsiness_model.pkl    # 訓練好的模型檔案 (由 train_model.py 生成)
├── model_meta.json         # 儲存模型相關元數據 (例如最佳化閾值)
│
├── requirements.txt        # 專案依賴的 Python 套件
├── data/                     # 訓練/測試用的圖片資料集 (未包含於 Git)
│   ├── drowsy/
│   └── notdrowsy/
│
├── training_data.csv       # 從 data/ 生成的原始特徵資料
├── training_data_clean.csv # 清理後的特徵資料
│
└── README.md               # 本說明檔案
```

## 安裝與執行

### 1. 環境設置

**前置需求**:
- Python 3.9+
- Git

**步驟**:

a. **克隆儲存庫**:
   ```bash
   git clone https://github.com/dses50117/final-project.git
   cd final-project
   ```

b. **建立虛擬環境 (建議)**:
   ```bash
   python -m venv .venv
   ```
   - 在 Windows 上啟用:
     ```powershell
     .\.venv\Scripts\Activate.ps1
     ```
   - 在 macOS/Linux 上啟用:
     ```bash
     source .venv/bin/activate
     ```

c. **安裝依賴套件**:
   ```bash
   pip install -r requirements.txt
   ```

### 2. 準備模型

在執行主應用程式之前，您需要先有 `drowsiness_model.pkl` 模型檔案。您可以透過以下兩種方式之一獲得：

**選項 A: 執行訓練腳本 (建議)**

如果您有 `data` 資料夾並想從頭訓練模型，請執行：
> **注意**: 訓練資料集 (`data` 目錄) 因體積過大未上傳至 GitHub。您需要自行準備符合格式的圖片資料。

```bash
# 1. 提取特徵 (將 data/ 內的圖片轉為特徵工程 CSV)
python feature_extraction.py

# 2. 訓練模型 (將使用 training_data_clean.csv 進行訓練)
python train_model.py
```
這將會在專案根目錄下生成 `drowsiness_model.pkl` 和 `model_meta.json`。

**選項 B: 手動下載**

如果您無法自行訓練，可以從其他來源取得已訓練好的 `drowsiness_model.pkl` 檔案，並將其放置在專案的根目錄下。

### 3. 啟動應用程式

確認 `drowsiness_model.pkl` 檔案存在後，執行以下命令啟動 Streamlit 應用：

```bash
streamlit run app.py
```

應用程式將在您的瀏覽器中開啟。您可以在側邊欄選擇 "📷 即時 Webcam" 或 "📂 影片檔案上傳" 模式，並調整偵測參數。

## 系統流程

1.  **影像輸入**: 從攝影機或影片檔案讀取影格。
2.  **人臉偵測**: 使用 MediaPipe Face Mesh 找出人臉的 478 個關鍵點。
3.  **特徵計算**:
    - 從眼睛關鍵點計算 **EAR**。
    - 從嘴部關鍵點計算 **MAR**。
    - 從頭部關鍵點估算 **Pitch, Yaw, Roll** 角度。
4.  **模型預測**: 將計算出的 6 個特徵 `[left_ear, right_ear, mar, pitch, yaw, roll]` 輸入預先訓練好的 XGBoost 模型，得到一個 "drowsy" 的機率。
5.  **狀態判斷**:
    - **模型狀態**: 如果 "drowsy" 機率高於設定的 `警示閾值`，且持續超過 `連續幀數`，則判定為疲勞。
    - **閉眼狀態**: 如果平均 EAR 低於 `閉眼 EAR 閾值`，且持續時間超過 `閉眼判定秒數`，則判定為疲勞。
6.  **發出警示**: 任一狀態滿足時，在畫面上顯示醒目的警示訊息。
7.  **結果顯示**: 在畫面上繪製人臉關鍵點，並在左下角顯示即時的疲勞機率、EAR值和閉眼秒數。
