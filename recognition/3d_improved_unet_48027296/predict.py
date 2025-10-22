import torch
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

    model = Improved3DUNet(in_channels=1, num_classes=1).cuda()
    model.load_state_dict(torch.load("improved_unet3d.pth"))
    model.eval()
    print("Model loaded and set to eval mode")

    os.makedirs("predictions_3d_improved", exist_ok=True)

    total_dice = 0.0

    for i in range(len(test_ds)):
        x, y = test_ds[i]
        with torch.no_grad():
            pred = model(x.unsqueeze(0).cuda()).cpu().squeeze(0)

        pred_bin = (torch.sigmoid(pred) > 0.5).float() # convert logits to probabilities
        dice = dice_score(pred_bin, y.cpu())
        total_dice += dice.item()
        print(f"Sample {i}: Dice = {dice:.4f}")

        if i % 10 == 0:
            plt.figure(figsize=(12, 4))

            mid_slice = x.shape[-1] // 2  # middle slice index

            plt.subplot(1, 3, 1)
            plt.imshow(x[0, :, :, mid_slice].cpu(), cmap='gray')
            plt.title("Input")

            plt.subplot(1, 3, 2)
            plt.imshow(y[0, :, :, mid_slice].cpu(), cmap='gray')
            plt.title("Ground Truth")

            plt.subplot(1, 3, 3)
            plt.imshow(pred_bin[0, :, :, mid_slice].cpu(), cmap='gray')
            plt.title("Prediction")

            plt.tight_layout()
            plt.savefig(f"predictions_3d_improved/sample_{i:03d}.png")
            plt.close()
            print(f"Saved predictions_3d_improved/sample_{i:03d}.png", flush=True)

if __name__ == "__main__":
    predict()
