import cv2
import pickle
import numpy as np
import mediapipe as mp
import tensorflow as tf

# =====================
# LOAD MODEL
# =====================

model = tf.keras.models.load_model(
    "models/landmark_model.keras"
)

with open(
    "models/labels.pkl",
    "rb"
) as f:

    encoder = pickle.load(f)

# =====================
# MEDIAPIPE
# =====================

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# =====================
# CAMERA
# =====================

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = hands.process(rgb)

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

            prediction = model.predict(
                x,
                verbose=0
            )

            idx = np.argmax(prediction)

            confidence = np.max(prediction)

            letter = encoder.inverse_transform(
                [idx]
            )[0]

            cv2.putText(
                frame,
                f"{letter} ({confidence:.2f})",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

        mp_draw.draw_landmarks(
            frame,
            hand,
            mp_hands.HAND_CONNECTIONS
        )

    cv2.imshow(
        "Sign Language Test",
        frame
    )

    key = cv2.waitKey(1)

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()