import torch
from modules import UNet2D
from dataset import Prostate2DDataset
import matplotlib.pyplot as plt
from utils import dice_score
import os

def predict():
    print("Starting predict.py...")

    model = UNet2D().cuda()
    model.load_state_dict(torch.load("unet2d.pth"))
    model.eval()
    print("Model loaded and set to eval mode")

    ds = Prostate2DDataset("/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_test",
                           "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_seg_test")
    print(f"Loaded {len(ds)} test samples")

    os.makedirs("predictions", exist_ok=True)

    total_dice = 0.0
    for i in range(len(ds)):
        x, y = ds[i]
        with torch.no_grad():
            pred = model(x.unsqueeze(0).cuda()).cpu().squeeze()

        pred_bin = (pred > 0.5).float()
        dice = dice_score(pred_bin, y.cpu())
        total_dice += dice.item()
        print(f"Sample {i}: Dice = {dice:.4f}")

        if i % 10 == 0:
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 3, 1); plt.imshow(x.squeeze().cpu(), cmap='gray'); plt.title("Input")
            plt.subplot(1, 3, 2); plt.imshow(y.squeeze().cpu(), cmap='gray'); plt.title("Ground Truth")
            plt.subplot(1, 3, 3); plt.imshow(pred_bin.squeeze(), cmap='gray'); plt.title("Prediction")
            plt.tight_layout()
            plt.savefig(f"predictions/sample_{i:03d}.png")
            plt.close()
            print(f"Saved predictions/sample_{i:03d}.png", flush=True)

if __name__ == "__main__":
    predict()
