import librosa
import numpy as np
import pickle

# Load preprocessed data
DATA_PATH = "/Users/pranav/Desktop/portfolio/musictutor/data/processed/mfcc_dataset.pkl"

with open(DATA_PATH, "rb") as f:
    data = pickle.load(f)

for note, mfcc in data.items():
    print(f"Note: {note}, MFCC Shape: {mfcc.shape}")
