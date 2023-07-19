import streamlit as st
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
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

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    if len(hands) == 1:  # Memproses satu tangan
        hand = hands[0]
        x, y, w, h = hand["bbox"]

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset : y + h + offset, x - offset : x + w + offset]

        if not imgCrop.size == 0:
            imgCropShape = imgCrop.shape
            aspectRatio = h / w

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                imgResize = cv2.resize(imgResize, (imgSize, imgSize))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((imgSize - imgResizeShape[0]) / 2)
                imgWhite[hGap : hGap + imgResizeShape[0], :] = imgResize

                if model_choice == "1":
                    # Menggunakan model AI untuk abjad
                    prediction, index = classifier1.getPrediction(imgWhite, draw=False)
                    label = labels1[index]
                else:
                    # Menggunakan model AI untuk kosakata
                    prediction, index = classifier2.getPrediction(imgWhite, draw=False)
                    label = labels2[index]

            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                imgResize = cv2.resize(imgResize, (imgSize, imgSize))
                imgResizeShape = imgResize.shape
                wGap = math.ceil((imgSize - imgResizeShape[1]) / 2)
                imgWhite[:, wGap : wGap + imgResizeShape[1]] = imgResize

                if model_choice == "1":
                    # Menggunakan model AI untuk abjad
                    prediction, index = classifier1.getPrediction(imgWhite, draw=False)
                    label = labels1[index]
                else:
                    # Menggunakan model AI untuk kosakata
                    prediction, index = classifier2.getPrediction(imgWhite, draw=False)
                    label = labels2[index]

            # Mendapatkan ukuran teks
            textSize = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX, 1.7, 2)[0]
            textWidth = textSize[0]
            textHeight = textSize[1]

            # Mengatur ukuran kotak berdasarkan ukuran teks
            boxWidth = textWidth + 40
            boxHeight = textHeight + 20

            cv2.rectangle(
                imgOutput,
                (x - offset, y - offset - boxHeight),
                (x - offset + boxWidth, y - offset),
                (0, 0, 255),
                cv2.FILLED,
            )  # Ubah warna menjadi merah
            cv2.putText(
                imgOutput,
                label,
                (x, y - 26),
                cv2.FONT_HERSHEY_COMPLEX,
                1.7,
                (255, 255, 255),
                2,
            )
            cv2.rectangle(
                imgOutput,
                (x - offset, y - offset),
                (x + w + offset, y + h + offset),
                (0, 0, 255),
                4,
            )  # Ubah warna menjadi merah

            cv2.imshow("ImageCrop", imgCrop)
            cv2.imshow("ImageWhite", imgWhite)

    elif len(hands) == 2:  # Memproses dua tangan
        hand1, hand2 = hands[0], hands[1]
        x1, y1, w1, h1 = hand1["bbox"]
        x2, y2, w2, h2 = hand2["bbox"]

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        x_min = min(x1, x2) - offset
        y_min = min(y1, y2) - offset
        x_max = max(x1 + w1, x2 + w2) + offset
        y_max = max(y1 + h1, y2 + h2) + offset

        imgCropCombined = img[y_min:y_max, x_min:x_max]

        if not imgCropCombined.size == 0:
            imgCropShape = imgCropCombined.shape
            aspectRatio = imgCropShape[0] / imgCropShape[1]

            if aspectRatio > 1:
                k = imgSize / imgCropShape[0]
                wCal = math.ceil(k * imgCropShape[1])
                imgResize = cv2.resize(imgCropCombined, (wCal, imgSize))
                imgResize = cv2.resize(imgResize, (imgSize, imgSize))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((imgSize - imgResizeShape[0]) / 2)
                imgWhite[hGap : hGap + imgResizeShape[0], :] = imgResize
            else:
                k = imgSize / imgCropShape[1]
                hCal = math.ceil(k * imgCropShape[0])
                imgResize = cv2.resize(imgCropCombined, (imgSize, hCal))
                imgResize = cv2.resize(imgResize, (imgSize, imgSize))
                imgResizeShape = imgResize.shape
                wGap = math.ceil((imgSize - imgResizeShape[1]) / 2)
                imgWhite[:, wGap : wGap + imgResizeShape[1]] = imgResize

            if model_choice == "1":
                # Menggunakan model AI untuk abjad
                prediction, index = classifier1.getPrediction(imgWhite, draw=False)
                label = labels1[index]
            else:
                # Menggunakan model AI untuk kosakata
                prediction, index = classifier2.getPrediction(imgWhite, draw=False)
                label = labels2[index]

            # Mendapatkan ukuran teks
            textSize = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX, 1.7, 2)[0]
            textWidth = textSize[0]
            textHeight = textSize[1]

            # Mengatur ukuran kotak berdasarkan ukuran teks
            boxWidth = textWidth + 40
            boxHeight = textHeight + 20

            cv2.rectangle(
                imgOutput,
                (x_min - offset, y_min - offset - boxHeight),
                (x_min - offset + boxWidth, y_min - offset),
                (0, 0, 255),
                cv2.FILLED,
            )  # Ubah warna menjadi merah
            cv2.putText(
                imgOutput,
                label,
                (x_min, y_min - 26),
                cv2.FONT_HERSHEY_COMPLEX,
                1.7,
                (255, 255, 255),
                2,
            )
            cv2.rectangle(
                imgOutput,
                (x_min - offset, y_min - offset),
                (x_max + offset, y_max + offset),
                (0, 0, 255),
                4,
            )  # Ubah warna menjadi merah

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
