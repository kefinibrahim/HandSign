import streamlit as st
import numpy as np
import math
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

st.title("SIBI TRANSLATOR")
st.write("Ini adalah aplikasi penerjemah bahasa isyarat berdasarkan SIBI")

genre = st.radio(
    "Model AI mana yang akan kamu gunakan?",
    ("Abjad", "Kosakata")
)

if genre == "Abjad":
    st.write("Anda memilih abjad")
    model_choice = "1"
elif genre == "Kosakata":
    st.write("Anda memilih Kosakata")
    model_choice = "2"

detector = HandDetector(maxHands=2)  # Mengaktifkan deteksi dua tangan
offset = 20
imgSize = 600  # Mengatur ukuran gambar

labels1 = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W",
           "X", "Y", "Z"]

labels2 = ["I love you", "Thanks", "Yes"]

# Load model AI untuk abjad
classifier1 = Classifier("Model/keras_model.h5", "Model/labels.txt")

# Load model AI untuk kosakata
classifier2 = Classifier("Model/keras_model1.h5", "Model/labels1.txt")

class VideoProcessor(VideoProcessorBase):
    def __init__(self) -> None:
        super().__init__()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        imgOutput = img.copy()
        hands, img = detector.findHands(img)

        # ... (sisa kode Anda)

        return av.VideoFrame.from_ndarray(imgOutput, format="bgr24")

# Configurasi RTC untuk mengakses kamera
rtc_configuration = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})


webrtc_streamer(key="example", video_processor_factory=VideoProcessor, rtc_configuration=rtc_configuration)
