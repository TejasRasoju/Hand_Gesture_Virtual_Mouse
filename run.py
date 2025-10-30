import cv2, mediapipe as mp, torch, numpy as np, pyautogui, json, os
from model import GestureMLP
from dataset import LandmarkSequenceDataset

mp_hands = mp.solutions.hands

# ===== Load label map =====
with open("models/label_map.json") as f:
    label_map = json.load(f)
num_classes = len(label_map)

# ===== Load model =====
model = GestureMLP(input_dim=63, num_classes=num_classes)
model.load_state_dict(torch.load("models/gesture_mlp.pt", map_location="cpu"))
model.eval()

# ===== Load scaler =====
ds = LandmarkSequenceDataset("data/hand_dataset.csv")
scaler = ds.scaler

def normalize_landmarks(coords):
    coords = np.array(coords).reshape(-1, 3)
    base = coords[0]
    coords = coords - base
    max_val = np.max(np.abs(coords))
    coords = coords / (max_val + 1e-6)
    return coords.flatten().tolist()

def handle_action(pred_label, index_finger_tip):
    screen_w, screen_h = pyautogui.size()
    ix, iy = index_finger_tip
    ix = 1 - ix  # Mirror horizontally

    if pred_label == "move":
        pyautogui.moveTo(int(ix * screen_w), int(iy * screen_h))
    elif pred_label == "left_click":
        pyautogui.click()
    elif pred_label == "right_click":
        pyautogui.rightClick()
    elif pred_label == "scroll_up":
        pyautogui.scroll(300)
    elif pred_label == "scroll_down":
        pyautogui.scroll(-300)

cap = cv2.VideoCapture(0)
hands = mp_hands.Hands(max_num_hands=1)

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
            coords = normalize_landmarks(coords)

            X = scaler.transform([coords])
            X = torch.tensor(X, dtype=torch.float32)
            with torch.no_grad():
                out = model(X)
                pred = torch.argmax(out).item()
            label = label_map[str(pred)]

            index_finger_tip = [hand_landmarks.landmark[8].x, hand_landmarks.landmark[8].y]
            handle_action(label, index_finger_tip)

            cv2.putText(frame, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Virtual Mouse", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
