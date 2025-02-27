import torch
import librosa
import sounddevice as sd
import numpy as np
import os
import torch.nn as nn

SAMPLE_RATE = 22050
DURATION = 3  # 3 seconds

# Ensure models directory exists
MODEL_PATH = "../models/bansuri_model.pth"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("🚨 Model file not found! Please run `model_training.py` first.")

# Define the same CNN Model used in training
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


# Load Trained Model
model = CNNModel()
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# Label Mapping (Must match training)
LABELS = ["Sa", "Re", "Ga", "Ma", "Pa", "Dha", "Ni", "Sa'"]

def record_and_classify():
    print("🎤 Recording...")
    audio = sd.rec(int(SAMPLE_RATE * DURATION), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()
    print("✅ Recording complete.")

    # Convert to MFCCs
    mfccs = librosa.feature.mfcc(y=np.squeeze(audio), sr=SAMPLE_RATE, n_mfcc=13)
    mfccs = torch.tensor(mfccs, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    # Predict Note
    with torch.no_grad():
        output = model(mfccs)
        predicted_index = torch.argmax(output).item()
        predicted_note = LABELS[predicted_index]

    print(f"🎶 Predicted Note: {predicted_note}")

if __name__ == "__main__":
    record_and_classify()
