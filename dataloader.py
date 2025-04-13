import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from pytorch_wavelets import DWT1DForward
from scipy.signal import butter, filtfilt
import torch.nn.functional as F

class LargeCSVDataset(Dataset):
    def __init__(self, csv_file, chunk_size=1000, augment=False, stat_scaler=None, wavelet_scaler=None):
        self.data = pd.read_csv(csv_file, chunksize=chunk_size)
        self.data = pd.concat(chunk for chunk in self.data)
        self.dwt = DWT1DForward(wave='haar', J=2)
        self.augment = augment
        self.stat_scaler = stat_scaler
        self.wavelet_scaler = wavelet_scaler

    def normalize(self, data, scaler):
        """
        Normalize data using mean and std provided in the scaler dictionary.
        """
        return (data - scaler['mean']) / scaler['std']

    def __len__(self):
        return len(self.data)

    def add_noise(self, signal, noise_level=0.9):
        noise = torch.randn_like(signal) * noise_level
        return signal + noise

    def scale_signal(self, signal, factor=1.2):
        return signal * factor

    def time_shift(self, signal, shift=10):
        return torch.roll(signal, shifts=shift, dims=0)

    def frequency_mask(self, signal, mask_fraction=0.1):
        fft_signal = torch.fft.fft(signal)
        num_to_mask = int(mask_fraction * len(fft_signal))
        indices_to_mask = torch.randperm(len(fft_signal))[:num_to_mask]
        fft_signal[indices_to_mask] = 0
        return torch.fft.ifft(fft_signal).real

    def frequency_shift(self, signal, shift=5):
        fft_signal = torch.fft.fft(signal)
        shifted_fft_signal = torch.roll(fft_signal, shifts=shift, dims=0)
        return torch.fft.ifft(shifted_fft_signal).real

    def frequency_warp(self, signal, warp_factor=1.1):
        fft_signal = torch.fft.fft(signal)
        warped_fft_signal = torch.fft.fftshift(fft_signal) * warp_factor
        return torch.fft.ifft(warped_fft_signal).real

    def bandpass_filter(self, signal, low_freq=0.1, high_freq=0.4, sample_rate=1.0):
        nyquist = 0.5 * sample_rate
        low = low_freq / nyquist
        high = high_freq / nyquist
        b, a = butter(4, [low, high], btype='band')
        filtered_signal = filtfilt(b, a, signal.cpu().numpy())
        return torch.tensor(filtered_signal, dtype=torch.float32)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        features_stat = torch.tensor(row[0:16].values, dtype=torch.float32)

        # Apply DWT and reshape
        features_signal, _ = self.dwt(torch.tensor(row[16:-1].values, dtype=torch.float32).view(1, 1, 2*750))
        features_signal = features_signal.view(-1)

        # Apply augmentations if specified
        if self.augment:
            features_signal = self.add_noise(features_signal)

        # Normalize the features
        if self.stat_scaler:
            features_stat = self.normalize(features_stat, self.stat_scaler)
        if self.wavelet_scaler:
            features_signal = self.normalize(features_signal, self.wavelet_scaler)

        label = torch.tensor(F.one_hot(torch.tensor(row[-1], dtype=torch.int64), num_classes=3), dtype=torch.float32)

        return features_stat, features_signal, label

def compute_scalers(dataset):
    """
    Compute mean and std for stat_features and wavelet_features across the dataset.
    """
    all_stat_features = []
    all_wavelet_features = []

    for idx in range(len(dataset)):
        stat_features, wavelet_features, _ = dataset[idx]
        all_stat_features.append(stat_features)
        all_wavelet_features.append(wavelet_features)

    all_stat_features = torch.stack(all_stat_features)
    all_wavelet_features = torch.stack(all_wavelet_features)

    stat_scaler = {
        'mean': all_stat_features.mean(dim=0),
        'std': all_stat_features.std(dim=0)
    }
    wavelet_scaler = {
        'mean': all_wavelet_features.mean(dim=0),
        'std': all_wavelet_features.std(dim=0)
    }

    return stat_scaler, wavelet_scaler
