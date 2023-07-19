import streamlit as st
import numpy as np
import math

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

# Mendapatkan ukuran frame video
cap = cv2.VideoCapture(0)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Membuat tampilan untuk video
video_placeholder = st.empty()

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

# Periksa apakah aplikasi berjalan di Streamlit Cloud atau platform serupa
try:
    import streamlit.ReportThread as ReportThread
    import streamlit.server.Server as Server

    # Streamlit Cloud atau platform serupa
    IS_STREAMLIT_CLOUD = hasattr(ReportThread, 'get_report_ctx') and hasattr(Server, '_get_server_details')
except Exception as e:
    # Lingkungan lokal
    IS_STREAMLIT_CLOUD = False

# Pilih paket OpenCV yang sesuai berdasarkan platform
if IS_STREAMLIT_CLOUD:
    import cv2
else:
    import cv2  # Ubah ini menjadi 'import cv2 as cv2_headless' jika Anda telah menginstal 'opencv-python-headless'

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    # ... (sisa kode Anda)

    # Menampilkan frame video di Streamlit
    video_placeholder.image(imgOutput, channels="BGR")

    key = cv2.waitKey(1)

    if key == ord("1"):
        model_choice = "1"
    elif key == ord("2"):
        model_choice = "2"
    elif key == 27:
        break

cap.release()
cv2.destroyAllWindows()
