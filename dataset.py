# dataset.py
import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle
import os

class LandmarkSequenceDataset(Dataset):
    def __init__(self, csv_file, sequence_len=1, scaler_path=None, save_scaler_path=None):
        df = pd.read_csv(csv_file)
        self.labels = df["label"].astype("category").cat.codes.values
        # label_map: index -> label string
        self.label_map = dict(enumerate(df["label"].astype("category").cat.categories))

        X = df.drop(columns=["label"]).values.astype("float32")
        # if scaler_path provided and exists, load; else fit and optionally save
        if scaler_path and os.path.exists(scaler_path):
            with open(scaler_path, "rb") as f:
                self.scaler = pickle.load(f)
        else:
            self.scaler = StandardScaler()
            self.scaler.fit(X)
            if save_scaler_path:
                with open(save_scaler_path, "wb") as f:
                    pickle.dump(self.scaler, f)

        X = self.scaler.transform(X)
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(self.labels, dtype=torch.long)
        self.sequence_len = sequence_len

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
