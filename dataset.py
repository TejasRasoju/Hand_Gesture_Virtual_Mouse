import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn.preprocessing import StandardScaler

class LandmarkSequenceDataset(Dataset):
    def __init__(self, csv_file, sequence_len=1):
        df = pd.read_csv(csv_file)
        self.labels = df["label"].astype("category").cat.codes.values
        self.label_map = dict(enumerate(df["label"].astype("category").cat.categories))

        X = df.drop(columns=["label"]).values.astype("float32")
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X)
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(self.labels, dtype=torch.long)
        self.sequence_len = sequence_len

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
