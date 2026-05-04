

import torch
import random
import torchaudio
from torch.utils.data import Dataset
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from typing import List, Dict

# ---------------------------------------
# MIMIIDataset
#
# Purpose:
#   - Loads, segments, and preprocesses raw .wav files from MIMII dataset
#   - Converts audio to mel-spectrograms for model input
#   - Applies optional augmentations and normalisation
#   - Supports sliding window segmentation for temporal modeling
#
# Output per item:
#   (mel[T, n_mels], y_anom, y_unit, file_path)
# ---------------------------------------

class MIMIIDataset(Dataset):
    """
    Pytorch dataset for MIMII audio anomaly detection
    
    Each record must include:
        path (Path), is_abnormal (int), unit_global_idx (int)
    """
    def __init__(self, records: List[Dict], sample_rate: int = 16000, n_fft: int = 1024, hop_length: int = 512, n_mels: int = 64, train: bool = True, target_seconds: float = 2.0, overlap: float = 0.5):
        self.records = records
        self.sample_rate = sample_rate
        self.train = train
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.target_seconds = target_seconds
        self.overlap = overlap
        
        # Sliding window segmentation across all audio files
        self._prepare_segments()

        # Mel spectrogram transform
        self.mel = torch.nn.Sequential(
            MelSpectrogram(
                sample_rate=sample_rate,
                n_fft = n_fft,
                hop_length = hop_length,
                n_mels=n_mels, 
                f_min = 50, 
                f_max = 8000
            ),
            AmplitudeToDB(stype = "power", top_db = 80))
        '''
        if train:
            self.aug = torch.nn.Sequential(
                torchaudio.transforms.FrequencyMasking(freq_mask_param = 8),
                torchaudio.transforms.TimeMasking(time_mask_param = 16)
            )
        else:
            self.aug = torch.nn.Identity()
        '''    
        # Augmentation settings
        if train:
            self.aug = torch.nn.Identity()  # Placeholder for futer masking/time augs
            self.extra_aug = False          # Can enable for extra augs
        else:
            self.aug = torch.nn.Identity()
            self.extra_aug = False
        
        # Placeholders
        self.resample = None
        self.global_mean = None
        self.global_std = None


    # Segment Preparation
    def _prepare_segments(self):
        """ Generate sliding window (start, end) indices for all files"""
        segments = []
        seg_len = int(self.target_seconds * self.sample_rate)
        hop_len = int(seg_len * (1 - self.overlap)) # OVERLAP
        
        for file_idx, r in enumerate(self.records):
            #Try get total fram length
            try:
                info = torchaudio.info(r["path"])
                total_len = int(info.num_frames)
            except Exception:
                wav, sr = torchaudio.load(r["path"])
                total_len = wav.size(1)
            
            #Short files padded to full window length with zeros
            if total_len < seg_len:
                segments.append((file_idx, 0, seg_len))
            else:
                # Slide window across full waveform with hop_len stride
                for start in range(0, total_len - seg_len + 1, hop_len):
                    segments.append((file_idx, start, start + seg_len))
                    
        self.segments = segments
        print(f"[SlidingWindow] Generated {len(segments)} segments from {len(self.records)} files "
              f"(hop = {hop_len / self.sample_rate:.1f}s, overlap={self.overlap:.2f}).")


    # Pytorch dataset interface
    def __len__(self): return len(self.segments)
    

    def __getitem__(self, i):
        """ Load one segment one convert to normalised mel spectrogram"""
        file_idx, start, end = self.segments[i]
        r = self.records[file_idx]
        wav, sr = torchaudio.load(r['path'])

        # Convert stereo to mono
        if wav.shape[0] > 1:  # mono-ise
            wav = wav.mean(dim=0, keepdim=True)

        # Resample if needed
        if sr != self.sample_rate:
            if self.resample is None or self.resample.orig_freq != sr:
                self.resample = torchaudio.transforms.Resample(sr, self.sample_rate)
            wav = self.resample(wav)

        # Crop or pad to segment length
        wav = wav[:, start:end]
        cur_len = wav.size(1)
        if cur_len < end - start:
            wav = torch.nn.functional.pad(wav, (0, (end - start) - cur_len))

        # Optional data augmentation for training
        if self.extra_aug:
            # Random gain scaling
            wav = wav * (0.8 + 0.4 * torch.rand(1))

            if torch.rand(1) < 0.5:
                # Inject random white noise
                if torch.rand(1) < 0.4:
                    noise_scale = random.uniform(0.001, 0.005)
                    wav = wav + noise_scale * torch.randn_like(wav)
                    
                # Apply small random time shift 
                if torch.rand(1) < 0.3:
                    shift = int(random.uniform(-0.03, 0.03) * self.sample_rate)
                    wav = torch.roll(wav, shifts= shift, dims = 1)
                    
                # Randomly mute short segment
                if torch.rand(1) < 0.2:
                    seg_len = int(0.02 * self.sample_rate)
                    start = random.randint(0, max(1, wav.size(1) - seg_len))
                    wav[:, start:start+seg_len] = 0.0


        # Mel spectrogram transformation
        mel = self.mel(wav).squeeze(0)
        
        # Normalisation (global preferred) otherwise instance level
        if self.global_mean is not None and self.global_std is not None:
            mel = (mel - self.global_mean[:, None]) / (self.global_std[:, None] + 1e-8)
        else:
            mel = (mel - mel.mean(dim = 1, keepdim = True)) / (mel.std(dim = 1, keepdim = True) + 1e-8)
      
        # Optional time/freq masking
        mel = self.aug(mel.unsqueeze(0)).squeeze(0)
        
        # Transpose so time is first dimension (T x n_mels)
        mel = mel.T.contiguous()    # [T, n_mels]

        if i == 0 and self.train:
            print(f"Sample mean = {mel.mean():.4f}, std={mel.std():.4f}")
            print(f"mel frames T = {mel.shape[0]} (hop = {self.hop_length})")

        # Labels
        y_anom    = torch.tensor(r['is_abnormal'], dtype=torch.long)
        y_unit    = torch.tensor(r['unit_global_idx'], dtype=torch.long)
        file_path = str(r['path'])

        return mel, y_anom, y_unit, file_path

    # Global Mean/std computation
    def compute_global_stats(self, use_normal_only = True):
        """ Compute dataset-wide mean and std for normalisation"""
        total_sum = torch.zeros(self.n_mels)
        total_sq_sum = torch.zeros(self.n_mels)
        total_frames = 0
        resample = None

        for idx, r in enumerate(self.records):
            # Optionally remove abnormal samples - leakage
            if use_normal_only and r["is_abnormal"]:
                continue

            wav, sr = torchaudio.load(r["path"])
            
            # standardise channel count and sample rate
            if wav.shape[0] > 1:
                wav = wav.mean(dim = 0, keepdim = True)
            if sr != self.sample_rate:
                if resample is None or resample.orig_freq != sr:
                    resample = torchaudio.transforms.Resample(sr, self.sample_rate)
                wav = resample(wav)

            # Convert to mel spectrogram and accumulate per-frequency stats
            mel = self.mel(wav).squeeze(0)
            total_sum += mel.sum(dim = 1)
            total_sq_sum += (mel ** 2).sum(dim = 1)
            total_frames += mel.size(1)

        # Weighted mean and std across all frames (channel wise)
        mean = total_sum / total_frames
        std = torch.sqrt(total_sq_sum / total_frames - mean ** 2).clamp_min(1e-5)
        print(f"Global (weighted) mean/std shapes: {mean.shape}, {std.shape}")
        return mean, std