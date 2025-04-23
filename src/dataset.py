import torch
from torch.utils.data import Dataset

class ECGDataset(Dataset):
    def __init__(self, signals, labels):
        """
        signals: numpy array shape [samples, length]
        labels: numpy array shape [samples]
        """
        self.signals = torch.tensor(signals, dtype=torch.float32).unsqueeze(1)  
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        return self.signals[idx], self.labels[idx]