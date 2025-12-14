方案 A：本機用 OpenCV 即時 Webcam（你應該已經會）

這是你現在在 VS Code / 本機跑得動的那套：

import cv2

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    # 處理 frame（丟 Mediapipe、畫框 etc.）


在 本機 跑 streamlit run app.py 時，這是 OK 的。
但只要部署到雲端 → 一律爆 Cannot read camera。

方案 B：在 Streamlit（包含雲端）用瀏覽器開相機（WebRTC：推薦）

要真正做到「在 Streamlit 頁面上，即時看到自己攝影機畫面，並且可以做 Mediapipe 處理」，
你可以用這個套件：

streamlit-webrtc（WebRTC 即時影音串流組件）

基本概念：

不是伺服器讀攝影機，而是「使用者的瀏覽器」開攝影機

視訊流透過 WebRTC 傳到你的 Python 程式

你在 callback 裡用 MediaPipe / OpenCV 做偵測，

再把處理後的影像回傳出去 → 就是即時 webcam demo。

安裝（放到你的 requirements.txt）：
streamlit-webrtc
av


av 是處理影像的底層套件。

超簡單範例（可以即時看到攝影機畫面）
import av
import cv2
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase


class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        # 這裡可以丟給 mediapipe 做偵測，例如臉部關鍵點、打瞌睡偵測等等
        # 這裡先簡單示範：轉成灰階
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_out = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
        return img_out


st.title("即時 Webcam Demo（streamlit-webrtc）")

webrtc_streamer(
    key="example",
    video_transformer_factory=VideoTransformer,
    media_stream_constraints={"video": True, "audio": False},
)


這個版本：

在本機跑可以

部署到 Streamlit Cloud / Render 也可以

因為實際開相機的是「使用者的瀏覽器」，不是伺服器

接下來你只要把你原本的 MediaPipe 臉部 / 疲勞偵測流程，搬到 VideoTransformer.transform() 裡，就可以變成 真正的雲端即時 webcam demo。