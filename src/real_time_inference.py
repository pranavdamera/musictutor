import librosa
import numpy as np
import torch
import torchaudio
import sounddevice as sd

def real_time_pitch_analysis(model, duration=5):
    print("Recording...")
    audio = sd.rec(int(22050 * duration), samplerate=22050, channels=1, dtype='float32')
    sd.wait()
    print("Recording Complete")

    mfccs = librosa.feature.mfcc(y=audio[:, 0], sr=22050, n_mfcc=13)
    mfccs = torch.tensor(mfccs).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        prediction = model(mfccs)

    print("Predicted Pitch Accuracy:", prediction)

if __name__ == "__main__":
    model = torch.load("../models/bansuri_model.pth")  # Load trained model
    real_time_pitch_analysis(model)
