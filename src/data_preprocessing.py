import librosa
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import librosa.display

# Paths
RAW_DATA_PATH = "data/raw/notes/"
PROCESSED_DATA_PATH = "data/processed/"
os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)  # Ensure processed folder exists

# Force all MFCCs to have the same number of time frames
FIXED_MFCC_SHAPE = (13, 50)  # (13 frequency bins, 50 time frames)

def extract_features(file_path):
    """Extract MFCC features from an audio file and standardize shape."""
    y, sr = librosa.load(file_path, sr=22050)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    # Pad or truncate MFCCs to match the required shape (13, 50)
    if mfccs.shape[1] < FIXED_MFCC_SHAPE[1]:  # If too short, pad with zeros
        pad_width = FIXED_MFCC_SHAPE[1] - mfccs.shape[1]
        mfccs = np.pad(mfccs, ((0, 0), (0, pad_width)), mode='constant')
    elif mfccs.shape[1] > FIXED_MFCC_SHAPE[1]:  # If too long, truncate
        mfccs = mfccs[:, :FIXED_MFCC_SHAPE[1]]

    return mfccs

def process_all_notes():
    """Extract MFCCs from all notes and save"""
    dataset = {}

    for file in os.listdir(RAW_DATA_PATH):
        if file.endswith(".wav"):
            note_name = file.replace(".wav", "")
            file_path = os.path.join(RAW_DATA_PATH, file)

            try:
                print(f"Processing: {note_name} ({file_path})")
                mfccs = extract_features(file_path)
                dataset[note_name] = mfccs  # Store extracted MFCCs

                # Save visualization
                plt.figure(figsize=(10, 4))
                librosa.display.specshow(mfccs, x_axis="time")
                plt.colorbar()
                plt.title(f"MFCC for {note_name}")
                plt.savefig(os.path.join(PROCESSED_DATA_PATH, f"{note_name}.png"))
                plt.close()
                print(f"Saved spectrogram for {note_name}")

            except Exception as e:
                print(f"Error processing {file}: {e}")

    # Save dataset as pickle file
    pickle_path = os.path.join(PROCESSED_DATA_PATH, "mfcc_dataset.pkl")
    with open(pickle_path, "wb") as f:
        pickle.dump(dataset, f)
    print(f"✅ Feature extraction completed. Data saved to {pickle_path}")

if __name__ == "__main__":
    process_all_notes()
