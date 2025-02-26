import sounddevice as sd
import numpy as np
import librosa
import scipy.io.wavfile as wav
import os

# Recording settings
SAMPLE_RATE = 22050  # Standard for librosa
DURATION = 5  # Record for 5 seconds

def record_audio(note_name):
    print(f"Recording {note_name} for {DURATION} seconds...")
    
    # Record audio
    audio = sd.rec(int(SAMPLE_RATE * DURATION), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()
    
    # Normalize audio
    audio = np.squeeze(audio)
    
    # Save as WAV file
    save_path = f"/Users/pranav/Desktop/portfolio/musictutor/data/raw/notes/{note_name}.wav"
    wav.write(save_path, SAMPLE_RATE, audio)
    
    print(f"Saved: {save_path}")

if __name__ == "__main__":
    note_name = input("Enter note name (Sa, Re, Ga, etc.): ")
    record_audio(note_name)
