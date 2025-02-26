import librosa
import numpy as np
import os
import matplotlib.pyplot as plt

def extract_features(file_path, save_spectrogram=False):
    y, sr = librosa.load(file_path, sr=22050)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    
    if save_spectrogram:
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(mfccs, x_axis="time")
        plt.colorbar()
        plt.title("MFCC")
        plt.savefig(file_path.replace(".wav", ".png"))

    return mfccs

if __name__ == "__main__":
    audio_path = "../data/raw/sample.wav"  # Replace with actual path
    if os.path.exists(audio_path):
        features = extract_features(audio_path, save_spectrogram=True)
        print("Extracted Features:", features.shape)
    else:
        print("Audio file not found. Please add bansuri recordings to data/raw/")
