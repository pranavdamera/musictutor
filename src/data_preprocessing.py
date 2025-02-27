import librosa
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt

# Paths
RAW_DATA_PATH = "/Users/pranav/Desktop/portfolio/musictutor/data/raw/notes"
PROCESSED_DATA_PATH = "/Users/pranav/Desktop/portfolio/musictutor/data/processed"

# Ensure processed folder exists
os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)

def extract_features(file_path):
    """Extract MFCC features from an audio file"""
    y, sr = librosa.load(file_path, sr=22050)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)  # Extract 13 MFCC coefficients
    return mfccs

def process_all_notes():
    """Extract MFCCs from all notes and save"""
    dataset = {}

    for file in os.listdir(RAW_DATA_PATH):
        if file.endswith(".wav"):
            note_name = file.replace(".wav", "")
            file_path = os.path.join(RAW_DATA_PATH, file)
            mfccs = extract_features(file_path)

            # Save as numpy array
            dataset[note_name] = mfccs

            # Save visualization
            plt.figure(figsize=(10, 4))
            librosa.display.specshow(mfccs, x_axis="time")
            plt.colorbar()
            plt.title(f"MFCC for {note_name}")
            plt.savefig(os.path.join(PROCESSED_DATA_PATH, f"{note_name}.png"))

    # Save dataset as pickle file
    with open(os.path.join(PROCESSED_DATA_PATH, "mfcc_dataset.pkl"), "wb") as f:
        pickle.dump(dataset, f)

    print("Feature extraction completed. Data saved!")

if __name__ == "__main__":
    process_all_notes()
