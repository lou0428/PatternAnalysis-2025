"""
    File name: predict.py
    Author: Louisa Wu
    Date created: 13/10/2025
    Date last modified: 23/10/2025
    Python Version: 3.9.23
    Description: 
"""

import torch
import numpy as np
from modules import Improved3DUNet
from dataset import Prostate3DDataset, get_data_splits
import matplotlib.pyplot as plt
from utils import dice_score
import os

def predict():
    print("Starting predict.py...")

    # directories 
    image_dir = "/home/groups/comp3710/HipMRI_Study_open/semantic_MRs"
    label_dir = "/home/groups/comp3710/HipMRI_Study_open/semantic_labels_only"

    _, _, _, _, test_imgs, test_lbls = get_data_splits(image_dir, label_dir)
    test_ds = Prostate3DDataset(test_imgs, test_lbls)
    print(f"Loaded {len(test_ds)} test samples")

    model = Improved3DUNet(in_channels=1, num_classes=6).cuda()
    model.load_state_dict(torch.load("improved_unet3d.pth"))
    model.eval()
    print("Model loaded and set to eval mode")

    os.makedirs("predictions_3d_improved", exist_ok=True)

    all_dice_scores = []

    for i in range(len(test_ds)):
        x, y = test_ds[i]
        with torch.no_grad():
            pred = model(x.unsqueeze(0).cuda()).cpu().squeeze(0)
            pred_labels = torch.argmax(pred, dim=0)

        dice_scores = dice_score(pred.unsqueeze(0), y.unsqueeze(0))
        all_dice_scores.append(dice_scores)
        print(f"Sample {i}: Dice per class = {[round(d, 4) for d in dice_scores]}")

        if i % 10 == 0:
            plt.figure(figsize=(12, 4))
            mid_slice = x.shape[-1] // 2

            plt.subplot(1, 3, 1)
            plt.imshow(x[0, :, :, mid_slice].cpu(), cmap='gray')
            plt.title("Input")

            plt.subplot(1, 3, 2)
            plt.imshow(y[:, :, mid_slice].cpu(), cmap='nipy_spectral', vmin=0, vmax=5)
            plt.title("Ground Truth")

            plt.subplot(1, 3, 3)
            plt.imshow(pred_labels[:, :, mid_slice].cpu(), cmap='nipy_spectral', vmin=0, vmax=5)
            plt.title("Prediction")

            plt.tight_layout()
            plt.savefig(f"predictions_3d_improved/sample_{i:03d}.png")
            plt.close()

    all_dice = np.array(all_dice_scores)  # shape: (N_samples, 6)
    mean_per_class = np.mean(all_dice, axis=0)

    plt.figure()
    plt.bar(range(6), mean_per_class)
    plt.xticks(range(6), ['Background', 'Body', 'Bones', 'Bladder', 'Rectum', 'Prostate'])
    plt.xlabel("Class")
    plt.ylabel("Mean Dice")
    plt.title("Mean Dice per Class (Test Set)")
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.savefig("test_dice_barplot.png")
    plt.close()

if __name__ == "__main__":
    predict()
