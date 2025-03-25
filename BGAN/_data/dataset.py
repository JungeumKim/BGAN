import torch
from torch.utils.data import Dataset, DataLoader

# Define custom PyTorch Dataset
class CustomDataset(Dataset):
    def __init__(self, Thetas_mat, X_mat):
        # Convert NumPy arrays to PyTorch tensors
        self.thetas = torch.from_numpy(Thetas_mat).float()
        self.x = torch.from_numpy(X_mat).float()

    def __len__(self):
        # Return the number of observations
        return len(self.thetas)

    def __getitem__(self, idx):
        # Return the Theta and X value for a specific index
        return self.thetas[idx], self.x[idx]

def infinite_loader(data_loader):
    """Generator to endlessly loop through the DataLoader."""
    while True:
        for x, y in data_loader:
            yield (x, y)
