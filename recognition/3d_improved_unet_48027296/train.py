print("train.py script started")

import torch
from torch.utils.data import DataLoader
from modules import UNet2D
from dataset import Prostate2DDataset
from utils import dice_score, plot_metrics
import torch.nn as nn
import torch.optim as optim
import os

def train_model():
    model = UNet2D().cuda()
    train_ds = Prostate2DDataset("/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_train",
                                 "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_seg_train")
    val_ds = Prostate2DDataset("/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_validate",
                               "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_seg_validate")
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=4)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    train_loss, val_dice = [], []

    print("Starting training...")
    print(f"Training on {len(train_loader)} batches")

    for epoch in range(20):
        model.train()
        epoch_loss = 0
        for x, y in train_loader:
            x, y = x.cuda(), y.cuda()
            pred = model(x)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        train_loss.append(epoch_loss / len(train_loader))

        model.eval()
        dice_total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.cuda(), y.cuda()
                pred = model(x)
                dice_total += dice_score(pred, y)
        val_dice.append(dice_total / len(val_loader))
        print(f"Epoch {epoch+1}: Loss={train_loss[-1]:.4f}, Dice={val_dice[-1]:.4f}")

    torch.save(model.state_dict(), "unet2d.pth")
    plot_metrics(train_loss, val_dice)

if __name__ == "__main__":
    train_model()