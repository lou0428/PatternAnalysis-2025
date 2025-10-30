"""
    File name: datasets.py
    Author: Louisa Wu
    Date created: 13/10/2025
    Date last modified: 31/10/2025
    Python Version: 3.9.23

    Description: 
        Defines PyTorch Dataset classes and helper functions for loading and preparing MRI 
        prostate datasets for 2D and 3D segmentation tasks.

        - Prostate2DDataset: handles 2D MRI slices
        - Prostate3DDataset: handles 3D volumetric MRI data
        - get_data_splits: splits data into training, validation, and test sets
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
    Dataloader for 2D MRI slice data. Loads and normalises NIfTI image and label files, resizes 
    them to a consistent spatial dimension, and applies optional random flipping. 
    """
    def __init__(self, image_dir, label_dir, img_size=256, random_flip=True):
        self.img_size = img_size
        self.random_flip = random_flip

        # collect and sort image and label paths 
        image_paths = sorted([
            os.path.join(image_dir, f) for f in os.listdir(image_dir)
            if f.endswith(('.nii', '.nii.gz'))
        ])
        label_paths = sorted([
            os.path.join(label_dir, f) for f in os.listdir(label_dir)
            if f.endswith(('.nii', '.nii.gz'))
        ])

        # ensure that the number of images and labels match 
        assert len(image_paths) == len(label_paths), "Mismatch between images and labels"

        print(f"Preloading {len(image_paths)} image-label pairs into memory...")
        start_time = time.time()

        self.data = []

        # load each image/label pair into memory 
        for i, (img_path, lbl_path) in enumerate(zip(image_paths, label_paths)):
            # load using nibabel 
            img = nib.load(img_path).get_fdata()
            lbl = nib.load(lbl_path).get_fdata()

            # handle cases where image has more than one slice 
            img = img[:, :, 0] if img.ndim == 3 else img
            lbl = lbl[:, :, 0] if lbl.ndim == 3 else lbl

            # normalise image intensity (zero mean with unit variance)
            img = (img - img.mean()) / img.std()

            # binarise label into foreground and background 
            lbl = (lbl > 0).astype(np.float32)

            # convert to PyTorch tensors and add channel dimension
            img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
            lbl = torch.tensor(lbl, dtype=torch.float32).unsqueeze(0)

            # resize both image and label to uniform spatial size 
            img = F.interpolate(img.unsqueeze(0), size=(img_size, img_size), mode='bilinear', align_corners=False).squeeze(0)
            lbl = F.interpolate(lbl.unsqueeze(0), size=(img_size, img_size), mode='nearest').squeeze(0)

            # store tuple (image, label)
            self.data.append((img, lbl))

            # display loading progress
            if (i + 1) % 100 == 0 or (i + 1) == len(image_paths):
                print(f"  Loaded {i + 1}/{len(image_paths)}")

        print(f"Preloading complete in {time.time() - start_time:.2f} seconds.")

    def __len__(self):
        """ Return the number of samples """
        return len(self.data)

    def __getitem__(self, idx):
        """ Return a image/label pair with optional random flips """
        img, lbl = self.data[idx]

        # apply random flipping augmentation
        if self.random_flip:
            if random.random() > 0.5:
                img = torch.flip(img, dims=[2]) # horizontal flip
                lbl = torch.flip(lbl, dims=[2])
            if random.random() > 0.5:
                img = torch.flip(img, dims=[1]) # vertical flip
                lbl = torch.flip(lbl, dims=[1])

        return img, lbl
    
class Prostate3DDataset(Dataset):
    """ 
    Dataloader for 3D prostate MRI volumes and their corresponding labels. 
    """
    def __init__(self, image_paths, label_paths, transform=None):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.transform = transform

    def __len__(self):
        """ Return the number of volumes """
        return len(self.image_paths)

    def __getitem__(self, idx):
        """ 
        Load, normalise and return a MRI volume/label pair.
        Images are standardised to zero mean and unit variance. 
        """
        # load MRI and label volumes 
        img = nib.load(self.image_paths[idx]).get_fdata()
        lbl = nib.load(self.label_paths[idx]).get_fdata()

        # normalise image intensity
        img = (img - np.mean(img)) / (np.std(img) + 1e-8)

        # convert to torch tensors
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)  # shape: (1, D, H, W)
        lbl = torch.tensor(lbl, dtype=torch.long)  # shape: (D, H, W)

        # apply custom transforms if provided 
        if self.transform:
            img, lbl = self.transform(img, lbl)

        return img, lbl

def get_data_splits(image_dir, label_dir, seed=42):
    """ 
    Splits all image/label pairs in a directory into training, validation and test sets using an 
    approximate 70/15/15 ratio. 
    """
    random.seed(seed) # uses a seed for reproducible shuffle 

    # collect sorted image and label file paths 
    image_paths = sorted([
        os.path.join(image_dir, f) 
        for f in os.listdir(image_dir) if f.endswith(('.nii', '.nii.gz'))
        ])
    label_paths = sorted([
        os.path.join(label_dir, f) 
        for f in os.listdir(label_dir) if f.endswith(('.nii', '.nii.gz'))
        ])

    assert len(image_paths) == len(label_paths), "Mismatch between images and labels"

    # shuffle indices of dataset 
    indices = list(range(len(image_paths)))
    random.shuffle(indices)

    # compute split sizes 
    n_total = len(indices)
    n_train = int(0.7 * n_total)
    n_val = int(0.15 * n_total)

    # split into training, validation, test sets 
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    # helper function to map indices to file paths 
    def subset(paths, idxs): 
        return [paths[i] for i in idxs]

    return (
        subset(image_paths, train_idx), subset(label_paths, train_idx),
        subset(image_paths, val_idx), subset(label_paths, val_idx),
        subset(image_paths, test_idx), subset(label_paths, test_idx)
    )
