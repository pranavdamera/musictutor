import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import numpy as np
import os
from torch.utils.data import DataLoader, Dataset

# Ensure models directory exists
os.makedirs("../models", exist_ok=True)

# Load preprocessed data
DATA_PATH = "/Users/pranav/Desktop/portfolio/musictutor/data/processed/mfcc_dataset.pkl"
with open(DATA_PATH, "rb") as f:
    data = pickle.load(f)

# Convert dataset into input-output format
X = []
y = []
label_mapping = {note: idx for idx, note in enumerate(data.keys())}  # Map notes to numbers

for note, mfcc in data.items():
    X.append(mfcc)
    y.append(label_mapping[note])

X = np.array(X)
y = np.array(y)

# Convert to Torch tensors
X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # Add channel dim
y = torch.tensor(y, dtype=torch.long)

# Define Dataset class
class BansuriDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Define Model
import torch
import torch.nn as nn

class CNNModel(nn.Module):
    def __init__(self, input_shape=(1, 13, 50)):  # Ensure input shape matches MFCC
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        
        # Dynamically determine flatten size
        self.flatten_size = self._get_flatten_size(input_shape)
        self.fc1 = nn.Linear(self.flatten_size, 8)  # 8 output classes (Sa Re Ga Ma...)

    def _get_flatten_size(self, input_shape):
        """Calculate the correct flatten size dynamically."""
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)  # Create dummy input
            dummy_output = self.conv1(dummy_input)  # Pass through convolution
            return dummy_output.numel()  # Compute flattened size

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)  # Flatten dynamically
        x = self.fc1(x)
        return x


# Initialize Model
model = CNNModel()

# Training Setup
dataset = BansuriDataset(X, y)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training Loop
for epoch in range(10):
    for batch_X, batch_y in dataloader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# Save the model
torch.save(model.state_dict(), "../models/bansuri_model.pth")
print("✅ Model training complete and saved!")
