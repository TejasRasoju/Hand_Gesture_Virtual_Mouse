import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from dataset import LandmarkSequenceDataset
from model import GestureMLP
import os, json

DATA_CSV = "data/hand_dataset.csv"
MODEL_PATH = "models"
os.makedirs(MODEL_PATH, exist_ok=True)

def train():
    ds = LandmarkSequenceDataset(DATA_CSV)
    num_classes = len(ds.label_map)  # auto-detect number of gestures
    print(f"Detected {num_classes} gesture classes: {list(ds.label_map.values())}")

    train_size = int(0.8 * len(ds))
    val_size = len(ds) - train_size
    train_ds, val_ds = random_split(ds, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GestureMLP(input_dim=63, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(30):
        model.train()
        total_loss = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: Train Loss = {total_loss/len(train_loader):.4f}")

    # Save model
    torch.save(model.state_dict(), os.path.join(MODEL_PATH, "gesture_mlp.pt"))

    # Save label map
    with open(os.path.join(MODEL_PATH, "label_map.json"), "w") as f:
        json.dump(ds.label_map, f)

    print(f"✅ Model and label map saved to '{MODEL_PATH}/'")

if __name__ == "__main__":
    train()
