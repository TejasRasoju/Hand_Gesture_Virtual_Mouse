import cv2
import mediapipe as mp
import pandas as pd
from pathlib import Path
import numpy as np

# Set up MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Paths
DATA_DIR = Path("data/hand_navigation_png")   # <-- folder containing gesture PNGs
OUT_CSV = Path("data/hand_dataset_from_png.csv")

# Assuming your images are in subfolders like: move/, left_click/, right_click/
gestures = [p.name for p in DATA_DIR.iterdir() if p.is_dir()]

rows = []

with mp_hands.Hands(static_image_mode=True, max_num_hands=1,
                    min_detection_confidence=0.5) as hands:
    for gesture in gestures:
        folder = DATA_DIR / gesture
        for img_path in folder.glob("*.png"):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)
            if not results.multi_hand_landmarks:
                continue

            hand = results.multi_hand_landmarks[0]
            coords = []
            for lm in hand.landmark:
                coords.extend([lm.x, lm.y])  # You can add lm.z later if you want depth info
            row = [gesture] + coords
            rows.append(row)

# Save to CSV
cols = ["label"] + [f"lm{i}" for i in range(42)]
df = pd.DataFrame(rows, columns=cols)
df.to_csv(OUT_CSV, index=False)
print(f"✅ Extracted {len(df)} samples from {len(gestures)} gestures.")
print(f"Saved landmark dataset to {OUT_CSV}")
