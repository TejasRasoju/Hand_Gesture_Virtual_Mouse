# collect.py
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import os, time, argparse
from utils import normalize_landmarks_xyz

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def collect(label, duration=15):
    csv_path = os.path.join(DATA_DIR, "hand_dataset.csv")
    cap = cv2.VideoCapture(0)
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

    start = time.time()
    rows = []
    print(f"Collecting '{label}' for {duration} seconds...")

    while time.time() - start < duration:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                coords = []
                for lm in hand_landmarks.landmark:
                    coords.extend([lm.x, lm.y, lm.z])
                norm_coords, _ = normalize_landmarks_xyz(coords)
                if norm_coords is None:
                    continue
                rows.append([label] + norm_coords)

                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.putText(frame, f"Gesture: {label}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.imshow("Collecting Data", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()

    if len(rows) == 0:
        print("No samples collected.")
        return

    df = pd.DataFrame(rows)
    header = ["label"] + [f"{a}_{i}" for i in range(21) for a in ("x","y","z")]
    df.columns = header
    if os.path.exists(csv_path):
        df.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        df.to_csv(csv_path, index=False)
    print(f"✅ Saved {len(rows)} samples to {csv_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, help="Gesture label (e.g. move, left_click, right_click)")
    ap.add_argument("--dur", type=int, default=15)
    args = ap.parse_args()
    collect(args.label, args.dur)
