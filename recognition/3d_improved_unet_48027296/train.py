print("train.py script started")

import torch
from torch.utils.data import DataLoader
from modules import ImprovedUNet3D
from dataset import Prostate3DDataset, get_data_splits
from utils import dice_score, plot_metrics
import torch.nn as nn
import torch.optim as optim
import os

def train_model():
    model = ImprovedUNet3D(in_channels=1, num_classes=1).cuda()

    # directories 
    image_dir = "/home/groups/comp3710/HipMRI_Study_open/semantic_MRs"
    label_dir = "/home/groups/comp3710/HipMRI_Study_open/semantic_labels_only"
    
    # split dataset 
    train_imgs, train_lbls, val_imgs, val_lbls, _test_imgs, _test_lbls = get_data_splits(image_dir, label_dir)
    train_ds = Prostate3DDataset(train_imgs, train_lbls)
    val_ds = Prostate3DDataset(val_imgs, val_lbls)

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1)

    criterion = nn.BCEWithLogitsLoss()
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

        # validation 
        model.eval()
        dice_total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.cuda(), y.cuda()
                pred = torch.sigmoid(model(x)) # convert logits to probabilities
                dice_total += dice_score(pred, y)
        val_dice.append(dice_total / len(val_loader))
        print(f"Epoch {epoch+1}: Loss={train_loss[-1]:.4f}, Dice={val_dice[-1]:.4f}")

    torch.save(model.state_dict(), "improved_unet3d.pth")
    plot_metrics(train_loss, val_dice)

if __name__ == "__main__":
    train_model()
    