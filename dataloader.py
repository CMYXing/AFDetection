import torch
from torch.utils.data import Dataset


class SupervisedLoader(Dataset):
    def __init__(self, data, label, transforms=None):
        self.data = data
        self.label = label
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        signal = self.data[idx]
        label = self.label[idx]

        if self.transforms is not None:
            signal = self.transforms(signal)

        return (torch.tensor(signal, dtype=torch.float),
                torch.tensor(label, dtype=torch.long))





