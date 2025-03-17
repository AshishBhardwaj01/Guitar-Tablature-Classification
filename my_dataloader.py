import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import os
from PIL import Image


class GuitarTabDataset(Dataset):
    def __init__(self, audio_dir, annotation_dir):
        self.audio_files = sorted([f for f in os.listdir(audio_dir) if f.endswith('.png')])
        self.annotation_files = sorted([f for f in os.listdir(annotation_dir) if f.endswith('.npy')])

        assert len(self.audio_files) == len(self.annotation_files), "Mismatch in audio and annotation file counts."

        self.audio_dir = audio_dir
        self.annotation_dir = annotation_dir

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        # Load spectrogram image
        audio_path = os.path.join(self.audio_dir, self.audio_files[idx])
        audio = Image.open(audio_path).convert("L")  # Convert to grayscale
        audio = np.array(audio, dtype=np.float32) / 255.0  # Normalize
        audio = torch.tensor(audio).unsqueeze(0)  # Shape: (1, H, W)

        # Load annotation file
        annotation_path = os.path.join(self.annotation_dir, self.annotation_files[idx])
        annotation = np.load(annotation_path, mmap_mode='r').astype(np.int64)  # Ensure correct type

        # Ensure annotation shape is (6,)
        annotation = torch.tensor(annotation)
        if annotation.shape[0] != 6:
            raise ValueError(f"Annotation file {annotation_path} has shape {annotation.shape}, expected (6,)")

        return audio, annotation

def create_dataloaders(audio_dir, annotation_dir, batch_size=32, train_ratio=0.8, val_ratio=0.1):
    dataset = GuitarTabDataset(audio_dir, annotation_dir)

    # Split dataset into training, validation, and testing
    train_size = int(train_ratio * len(dataset))
    val_size = int(val_ratio * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # Optimize DataLoader for CPU-based training
    loader_args = {
        'batch_size': batch_size,
        # 'num_workers': os.cpu_count() // 2,  # Use half of available CPU cores
        'num_workers': 2,  # Use half of available CPU cores
        'pin_memory': True,                 # Avoid pinning memory on CPU
        'prefetch_factor': 4,                # Preload batches
    }

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_args)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_args)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_args)

    return train_loader, val_loader, test_loader
