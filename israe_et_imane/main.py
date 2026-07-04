import sys
import cv2
import pickle
import threading
import numpy as np
import mediapipe as mp
import tensorflow as tf
import pyttsx3
from PyQt5.QtGui import (
    QImage,
    QPixmap
)
from PyQt5.QtWidgets import (
    QListWidget,
    QListWidgetItem
)

from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import (
    Qt,
    QTimer
)
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QHBoxLayout
)

from PyQt5.QtCore import Qt


class SignTranslator(QWidget):

    def __init__(self):

        super().__init__()
        # MODEL

        self.model = tf.keras.models.load_model(
            "models/landmark_model.keras"
        )

        with open(
            "models/labels.pkl",
            "rb"
        ) as f:

            self.encoder = pickle.load(f)

        # MEDIAPIPE

        self.mp_hands = mp.solutions.hands

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )


        self.mp_draw = mp.solutions.drawing_utils
        self.word = ""

        self.last_letter = ""
        self.stable_count = 0
        self.confirmed_letter = ""
 

        self.cap = cv2.VideoCapture(0)

        self.timer = QTimer()

        self.timer.timeout.connect(
            self.update_frame
        )

        self.timer.start(50)
        self.setWindowTitle(
            "AI Sign Language Translator"
        )

        self.resize(1400, 850)

        self.setStyleSheet("""
        QWidget{
            background-color:#1e1e1e;
            color:white;
            font-size:14px;
        }

        QTextEdit{
            background-color:#2d2d2d;
            color:white;
            border:2px solid #444;
        }

        QPushButton{
            background-color:#0078d7;
            color:white;
            border-radius:8px;
            padding:8px;
        }

        QPushButton:hover{
            background-color:#0094ff;
        }

        QListWidget{
            background-color:#2d2d2d;
            color:white;
        }
        """)


        self.word = ""

        self.setup_ui()
        
        self.alphabet_list.itemClicked.connect(
            self.show_sign_image
        )
        self.clear_btn.clicked.connect(
            self.clear_text
        )

        self.delete_btn.clicked.connect(
            self.delete_letter
        )

        self.speak_btn.clicked.connect(
            self.speak_text
        )

    def setup_ui(self):

        main_layout = QHBoxLayout()
        menu_layout = QVBoxLayout()
        
        menu_title = QLabel("📚 Alphabet")

        menu_title.setAlignment(Qt.AlignCenter)

        menu_layout.addWidget(menu_title)

        self.alphabet_list = QListWidget()

        letters = [
            "A","B","C","D","E","F","G","H",
            "I","K","L","M","N","O","P","Q",
            "R","S","T","U","V","W","X","Y"
        ]

        for letter in letters:

            self.alphabet_list.addItem(letter)

        menu_layout.addWidget(
            self.alphabet_list
        )

        left_layout = QVBoxLayout()

        title = QLabel(
            "🤟 Sign Translator"
        )

        title.setAlignment(
            Qt.AlignCenter
        )

        title.setStyleSheet(
            "font-size:24px;font-weight:bold;"
        )

        left_layout.addWidget(title)

        self.prediction_label = QLabel(
            "LETTER\n\n-"
        )

        self.prediction_label.setAlignment(
            Qt.AlignCenter
        )

        self.prediction_label.setStyleSheet(
            "font-size:28px;font-weight:bold;"
        )

        left_layout.addWidget(
            self.prediction_label
        )

        self.text_area = QTextEdit()

        left_layout.addWidget(
            self.text_area
        )

        self.speak_btn = QPushButton(
            "🔊 Speak"
        )

        left_layout.addWidget(
            self.speak_btn
        )

        self.delete_btn = QPushButton(
            "⌫ Delete"
        )

        left_layout.addWidget(
            self.delete_btn
        )

        self.clear_btn = QPushButton(
            "🗑 Clear"
        )

        left_layout.addWidget(
            self.clear_btn
        )

        # =====================
        # CAMERA
        # =====================

        camera_layout = QVBoxLayout()
        self.sign_image = QLabel()

        self.sign_image.setAlignment(
            Qt.AlignCenter
        )

        self.sign_image.setFixedHeight(250)
        self.sign_image = QLabel()

        self.sign_image.setAlignment(
            Qt.AlignCenter
        )

        self.sign_image.setFixedHeight(250)

        self.video_label = QLabel(
            "Camera"
        )

        self.video_label.setAlignment(
            Qt.AlignCenter
        )

        self.video_label.setStyleSheet(
            """
            background:black;
            color:white;
            font-size:20px;
            """
        )

        camera_layout.addWidget(
            self.video_label
        )

        camera_layout.addWidget(
            self.sign_image
        )

        main_layout.addLayout(
            menu_layout,
            1
        )

        main_layout.addLayout(
            left_layout,
            1
        )

        main_layout.addLayout(
            camera_layout,
            2
        )

        self.setLayout(
            main_layout
        )
    def update_frame(self):

        ret, frame = self.cap.read()

        if not ret:
            return

        frame = cv2.flip(
            frame,
            1
        )

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = self.hands.process(
            rgb
        )

        if results.multi_hand_landmarks:

            hand = results.multi_hand_landmarks[0]

            landmarks = []

            for lm in hand.landmark:

                landmarks.append(lm.x)
                landmarks.append(lm.y)

            if len(landmarks) == 42:

                x = np.array(
                    landmarks,
                    dtype=np.float32
                ).reshape(1, -1)

                prediction = self.model.predict(
                    x,
                    verbose=0
                )

                idx = np.argmax(
                    prediction
                )

                confidence = np.max(
                    prediction
                )

                letter = (
                    self.encoder
                    .inverse_transform([idx])[0]
                )

                letter = letter.replace(
                    "-samples",
                    ""
                )
                # AUTO TYPING

                if letter == self.last_letter:

                    self.stable_count += 1

                else:

                    self.stable_count = 0
                    self.last_letter = letter


                # AUTO TYPING

                if letter == self.last_letter:

                    self.stable_count += 1

                else:

                    self.last_letter = letter
                    self.stable_count = 0


                if (
                    confidence > 0.90
                    and self.stable_count > 10
                ):

                    if not self.word.endswith(letter):

                        self.word += letter

                        self.text_area.setText(
                            self.word
                        )

                self.prediction_label.setText(
                    f"{letter}\n{confidence:.2f}"
                )

            self.mp_draw.draw_landmarks(
                frame,
                hand,
                self.mp_hands.HAND_CONNECTIONS
            )

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        h, w, ch = rgb.shape

        image = QImage(
            rgb.data,
            w,
            h,
            ch * w,
            QImage.Format_RGB888
        )

        self.video_label.setPixmap(
            QPixmap.fromImage(image)
        )

    def clear_text(self):

        self.word = ""

        self.text_area.clear()


    def delete_letter(self):

        self.word = self.word[:-1]

        self.text_area.setText(
            self.word
        )


    def speak_text(self):

        text = self.text_area.toPlainText()

        if not text:
            return

        def worker():

            try:

                engine = pyttsx3.init()

                engine.say(text)

                engine.runAndWait()

                engine.stop()

            except Exception as e:

                print(e)

        threading.Thread(
            target=worker,
            daemon=True
        ).start()


    def show_sign_image(self, item):

        letter = item.text()

        image_path = f"alphabet/{letter}.png"

        print(image_path)

        pixmap = QPixmap(image_path)

        print("isNull =", pixmap.isNull())

        self.sign_image.setPixmap(
            pixmap.scaled(
                250,
                250,
                Qt.KeepAspectRatio
            )
        )
    def closeEvent(self, event):

            self.cap.release()

            event.accept()
if __name__ == "__main__":

    app = QApplication(
        sys.argv
    )

    window = SignTranslator()

    window.show()

    sys.exit(
        app.exec_()
    )