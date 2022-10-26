import torch


class Dataset:
    def __init__(self, data, targets) -> None:
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        label = self.targets[idx]