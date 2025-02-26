import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
from torch.utils.data import DataLoader, Dataset

class BansuriDataset(Dataset):
    def __init__(self, feature_files):
        self.data = [torch.load(f) for f in feature_files]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(32 * 10 * 10, 10)  # Adjust based on MFCC shape

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

if __name__ == "__main__":
    model = CNNModel()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Placeholder dataset
    dataset = BansuriDataset([])  # Replace with actual data loading
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    for epoch in range(5):
        for batch in dataloader:
            optimizer.zero_grad()
            outputs = model(batch)
            loss = criterion(outputs, torch.ones(len(batch), dtype=torch.long))
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {loss.item()}")
