"""
    File name: datasets.py
    Author: Louisa Wu
    Date created: 13/10/2025
    Date last modified: 23/10/2025
    Python Version: 3.9.23
    Description: 
"""

import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import os
import random
import time

class Prostate2DDataset(Dataset):
    """
    Dataloader for 2D datasets
    """
    def __init__(self, image_dir, label_dir, img_size=256, random_flip=True):
        self.img_size = img_size
        self.random_flip = random_flip

        image_paths = sorted([
            os.path.join(image_dir, f) for f in os.listdir(image_dir)
            if f.endswith(('.nii', '.nii.gz'))
        ])
        label_paths = sorted([
            os.path.join(label_dir, f) for f in os.listdir(label_dir)
            if f.endswith(('.nii', '.nii.gz'))
        ])
        assert len(image_paths) == len(label_paths), "Mismatch between images and labels"

        print(f"Preloading {len(image_paths)} image-label pairs into memory...")
        start_time = time.time()

        self.data = []
        for i, (img_path, lbl_path) in enumerate(zip(image_paths, label_paths)):
            img = nib.load(img_path).get_fdata()
            lbl = nib.load(lbl_path).get_fdata()

            img = img[:, :, 0] if img.ndim == 3 else img
            lbl = lbl[:, :, 0] if lbl.ndim == 3 else lbl

            img = (img - img.mean()) / img.std()
            lbl = (lbl > 0).astype(np.float32)

            img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
            lbl = torch.tensor(lbl, dtype=torch.float32).unsqueeze(0)

            img = F.interpolate(img.unsqueeze(0), size=(img_size, img_size), mode='bilinear', align_corners=False).squeeze(0)
            lbl = F.interpolate(lbl.unsqueeze(0), size=(img_size, img_size), mode='nearest').squeeze(0)

            self.data.append((img, lbl))

            if (i + 1) % 100 == 0 or (i + 1) == len(image_paths):
                print(f"  Loaded {i + 1}/{len(image_paths)}")

        print(f"Preloading complete in {time.time() - start_time:.2f} seconds.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, lbl = self.data[idx]

        if self.random_flip:
            if random.random() > 0.5:
                img = torch.flip(img, dims=[2])
                lbl = torch.flip(lbl, dims=[2])
            if random.random() > 0.5:
                img = torch.flip(img, dims=[1])
                lbl = torch.flip(lbl, dims=[1])

        return img, lbl
    
class Prostate3DDataset(Dataset):
    """ Dataloader for 3D prostate MRI valumes """
    def __init__(self, image_paths, label_paths, transform=None):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = nib.load(self.image_paths[idx]).get_fdata()
        lbl = nib.load(self.label_paths[idx]).get_fdata()

        # Normalize image
        img = (img - np.mean(img)) / (np.std(img) + 1e-8)

        # Convert to torch tensors
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)  # (1, D, H, W)
        lbl = torch.tensor(lbl, dtype=torch.long)  # (D, H, W)

        if self.transform:
            img, lbl = self.transform(img, lbl)

        return img, lbl

def get_data_splits(image_dir, label_dir, seed=42):
    """ Splits data into train/val/test set """
    random.seed(seed)
    image_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.nii', '.nii.gz'))])
    label_paths = sorted([os.path.join(label_dir, f) for f in os.listdir(label_dir) if f.endswith(('.nii', '.nii.gz'))])

    assert len(image_paths) == len(label_paths), "Mismatch between images and labels"
    indices = list(range(len(image_paths)))
    random.shuffle(indices)

    n_total = len(indices)
    n_train = int(0.7 * n_total)
    n_val = int(0.15 * n_total)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    def subset(paths, idxs): return [paths[i] for i in idxs]

    return (
        subset(image_paths, train_idx), subset(label_paths, train_idx),
        subset(image_paths, val_idx), subset(label_paths, val_idx),
        subset(image_paths, test_idx), subset(label_paths, test_idx)
    )
