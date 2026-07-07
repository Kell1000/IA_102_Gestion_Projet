import os
import csv
import cv2
import mediapipe as mp

DATASET_DIR = "dataset"
OUTPUT_CSV = "data/landmarks.csv"

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)

os.makedirs("data", exist_ok=True)

header = ["label"]

for i in range(21):
    header.append(f"x{i}")
    header.append(f"y{i}")

with open(OUTPUT_CSV, "w", newline="") as f:

    writer = csv.writer(f)
    writer.writerow(header)

    total = 0

    for class_name in os.listdir(DATASET_DIR):

        class_path = os.path.join(
            DATASET_DIR,
            class_name
        )

        if not os.path.isdir(class_path):
            continue

        print(f"Processing {class_name}")

        for img_name in os.listdir(class_path):

            img_path = os.path.join(
                class_path,
                img_name
            )

            image = cv2.imread(img_path)

            if image is None:
                continue

            rgb = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB
            )

            results = hands.process(rgb)

            if not results.multi_hand_landmarks:
                continue

            hand_landmarks = (
                results.multi_hand_landmarks[0]
            )

            row = [class_name]

            for lm in hand_landmarks.landmark:

                row.append(lm.x)
                row.append(lm.y)

            writer.writerow(row)

            total += 1

print(f"\n✅ Saved {total} samples")
print(f"✅ CSV created: {OUTPUT_CSV}")