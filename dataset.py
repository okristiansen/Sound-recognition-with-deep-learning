import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchaudio.transforms as T
import librosa
import numpy as np

class ESC50Dataset2(Dataset):
    def __init__(self, csv_path, audio_dir, folds, augment=False):
        self.data = pd.read_csv(csv_path)
        self.data = self.data[self.data["fold"].isin(folds)]
        self.audio_dir = audio_dir
        self.augment = augment
        self.freq_mask = T.FrequencyMasking(freq_mask_param=20)
        self.time_mask = T.TimeMasking(time_mask_param=40)

        self.spectrograms = []
        self.labels = []
        print("Loading and converting audio files...")
        for idx in range(len(self.data)):
            row = self.data.iloc[idx]
            file_path = os.path.join(self.audio_dir, row["filename"])
            y, sr = librosa.load(file_path, sr=22050)
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-6)
            mel_spec_db = torch.tensor(mel_spec_db).unsqueeze(0).float()
            self.spectrograms.append(mel_spec_db)
            self.labels.append(torch.tensor(row["target"]).long())
        print(f"Loaded {len(self.spectrograms)} samples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        mel_spec_db = self.spectrograms[idx]
        if self.augment:
            mel_spec_db = self.freq_mask(mel_spec_db)
            mel_spec_db = self.time_mask(mel_spec_db)
        return mel_spec_db, self.labels[idx]
     


def audio_to_melspectrogram(file_path):
     #y = waveform, sr = samplerate
    y, sr =librosa.load(file_path, sr=22050)

   
    #Convert to mel spectrogram
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels =128)

    #Convert to log scale
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    return mel_spec_db
