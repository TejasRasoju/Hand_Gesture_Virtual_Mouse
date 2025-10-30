# train.py
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from dataset import LandmarkSequenceDataset
from model import GestureMLP
import os, json, pickle
import numpy as np

DATA_CSV = "data/hand_dataset.csv"
MODEL_PATH = "models"
os.makedirs(MODEL_PATH, exist_ok=True)

def train():
    # Fit scaler and save it
    scaler_path = os.path.join(MODEL_PATH, "scaler.pkl")
    ds = LandmarkSequenceDataset(DATA_CSV, save_scaler_path=scaler_path)
    num_classes = len(ds.label_map)  # auto-detect number of gestures
    print(f"Detected {num_classes} gesture classes: {list(ds.label_map.values())}")

    train_size = int(0.8 * len(ds))
    val_size = len(ds) - train_size
    train_ds, val_ds = random_split(ds, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = ds.X.shape[1]
    model = GestureMLP(input_dim=input_dim, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_val_loss = float("inf")
    patience = 5
    patience_ctr = 0
    epochs = 30

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / max(1, len(train_loader))

        # validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for Xv, yv in val_loader:
                Xv, yv = Xv.to(device), yv.to(device)
                outv = model(Xv)
                lossv = criterion(outv, yv)
                val_loss += lossv.item()
                preds = torch.argmax(outv, dim=1)
                correct += (preds == yv).sum().item()
                total += yv.size(0)
        avg_val_loss = val_loss / max(1, len(val_loader))
        val_acc = 100.0 * correct / max(1, total)

        print(f"Epoch {epoch+1}/{epochs}  Train Loss={avg_train_loss:.4f}  Val Loss={avg_val_loss:.4f}  Val Acc={val_acc:.2f}%")

        # checkpoint on improvement
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_ctr = 0
            torch.save(model.state_dict(), os.path.join(MODEL_PATH, "gesture_mlp.pt"))
            # save label map (string keys)
            label_map_str_keys = {str(k): v for k, v in ds.label_map.items()}
            with open(os.path.join(MODEL_PATH, "label_map.json"), "w") as f:
                json.dump(label_map_str_keys, f)
            print(f"  -> Saved best model (val loss {best_val_loss:.4f})")
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print("Early stopping triggered.")
                break

    # Make sure scaler and label map exist (in case training finished without improvement)
    if not os.path.exists(os.path.join(MODEL_PATH, "gesture_mlp.pt")):
        torch.save(model.state_dict(), os.path.join(MODEL_PATH, "gesture_mlp.pt"))
        label_map_str_keys = {str(k): v for k, v in ds.label_map.items()}
        with open(os.path.join(MODEL_PATH, "label_map.json"), "w") as f:
            json.dump(label_map_str_keys, f)

    # scaler already saved in dataset init when save_scaler_path used
    print(f"✅ Model, label map, and scaler saved to '{MODEL_PATH}/'")

if __name__ == "__main__":
    train()
