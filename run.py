# run.py
import cv2, mediapipe as mp, torch, numpy as np, pyautogui, json, os, time, pickle
from model import GestureMLP
from dataset import LandmarkSequenceDataset
from utils import normalize_landmarks_xyz

mp_hands = mp.solutions.hands

MODEL_DIR = "models"
LABEL_MAP_PATH = os.path.join(MODEL_DIR, "label_map.json")
MODEL_PATH = os.path.join(MODEL_DIR, "gesture_mlp.pt")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

# ===== Load label map =====
if not os.path.exists(LABEL_MAP_PATH):
    raise FileNotFoundError("label_map.json not found in models/ — train the model first.")
with open(LABEL_MAP_PATH) as f:
    label_map = json.load(f)  # keys are strings "0","1",...
# reverse mapping: int -> label
label_map = {int(k): v for k, v in label_map.items()}
num_classes = len(label_map)

# ===== Load scaler =====
scaler = None
if os.path.exists(SCALER_PATH):
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
else:
    # Fallback: create dataset to get scaler (not ideal)
    print("Warning: scaler not found in models/. Falling back to dataset scaler fitted on CSV (not recommended).")
    ds = LandmarkSequenceDataset("data/hand_dataset.csv", save_scaler_path=None)
    scaler = ds.scaler

input_dim = scaler.mean_.shape[0]

# ===== Load model =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GestureMLP(input_dim=input_dim, num_classes=num_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# smoothing & debounce state
prev_pos = None
alpha = 0.25  # smoothing factor for mouse movement (0..1)
last_left_click = 0.0
last_right_click = 0.0
click_cooldown = 0.35  # seconds

def handle_action(pred_label, index_finger_tip):
    global prev_pos, last_left_click, last_right_click
    screen_w, screen_h = pyautogui.size()
    ix, iy = index_finger_tip
    ix = 1 - ix  # mirror horizontally if needed

    if pred_label == "move":
        # Smooth the movement
        x = int(ix * screen_w)
        y = int(iy * screen_h)
        if prev_pos is None:
            new_x, new_y = x, y
        else:
            new_x = int(prev_pos[0] * (1 - alpha) + x * alpha)
            new_y = int(prev_pos[1] * (1 - alpha) + y * alpha)
        pyautogui.moveTo(new_x, new_y)
        prev_pos = (new_x, new_y)

    elif pred_label == "left_click":
        now = time.time()
        if now - last_left_click > click_cooldown:
            pyautogui.click()
            last_left_click = now

    elif pred_label == "right_click":
        now = time.time()
        if now - last_right_click > click_cooldown:
            pyautogui.rightClick()
            last_right_click = now

    elif pred_label == "scroll_up":
        pyautogui.scroll(300)
    elif pred_label == "scroll_down":
        pyautogui.scroll(-300)

cap = cv2.VideoCapture(0)
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5)

while True:
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
            # normalize with utils
            norm_coords, index_xy = normalize_landmarks_xyz(coords)
            if norm_coords is None:
                continue

            X = scaler.transform([norm_coords])
            X = torch.tensor(X, dtype=torch.float32).to(device)
            with torch.no_grad():
                out = model(X)
                pred = torch.argmax(out, dim=1).item()
            label = label_map[pred]

            index_finger_tip = [hand_landmarks.landmark[8].x, hand_landmarks.landmark[8].y]
            handle_action(label, index_finger_tip)

            cv2.putText(frame, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    else:
        # no hands: optionally reset prev_pos so next move snaps? keep prev_pos to avoid jump
        pass

    cv2.imshow("Virtual Mouse", frame)
    # press 'q' or ESC to quit
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break

cap.release()
cv2.destroyAllWindows()
