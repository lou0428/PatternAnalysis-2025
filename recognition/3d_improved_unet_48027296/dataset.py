import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import os
import random
import time

class Prostate2DDataset(Dataset):
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