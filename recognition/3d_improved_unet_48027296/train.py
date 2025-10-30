"""
    File name: train.py
    Author: Louisa Wu
    Date created: 13/10/2025
    Date last modified: 31/10/2025
    Python Version: 3.9.23

    Description: 
        Training script for the Improved 3D UNet model on the HipMRI Prostate dataset.
        Handles model training, validation, loss and Dice score tracking and plotting results.
"""

import torch
from torch.utils.data import DataLoader
from modules import Improved3DUNet
from dataset import Prostate3DDataset, get_data_splits
from utils import dice_score, plot_metrics
import torch.nn as nn
import torch.optim as optim

def train_model():
    """ Trains and validates the Improved 3D UNet model on prostate MRI data """

    # model setup 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Improved3DUNet(in_channels=1, num_classes=6).to(device)

    # dataset setup 
    # directories 
    image_dir = "/home/groups/comp3710/HipMRI_Study_open/semantic_MRs"
    label_dir = "/home/groups/comp3710/HipMRI_Study_open/semantic_labels_only"
    
    # split dataset 
    train_imgs, train_lbls, val_imgs, val_lbls, _test_imgs, _test_lbls = get_data_splits(image_dir, label_dir)
    train_ds = Prostate3DDataset(train_imgs, train_lbls)
    val_ds = Prostate3DDataset(val_imgs, val_lbls)

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1)

    # loss, optimiser and metrics 
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 20
    num_classes = 6

    train_loss, val_loss, val_dice = [], [], []
    val_dice_per_class = [[] for _ in range(num_classes)]

    # training loop 
    print("Starting training...")
    print(f"Training on {len(train_loader)} batches")

    for epoch in range(num_epochs):
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

        # validation loop 
        model.eval()
        val_epoch_loss = 0
        dice_totals = [0.0] * num_classes

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.cuda(), y.cuda()
                pred = model(x)
                loss = criterion(pred, y)
                val_epoch_loss += loss.item()

                # compute Dice for each class 
                dice_scores = dice_score(pred, y)
                for c in range(num_classes):
                    dice_totals[c] += dice_scores[c]

        val_loss.append(val_epoch_loss / len(val_loader))
        val_dice.append(sum(dice_totals) / num_classes / len(val_loader))
        for c in range(num_classes):
            val_dice_per_class[c].append(dice_totals[c] / len(val_loader))
        print(f"Epoch {epoch+1}: Loss={train_loss[-1]:.4f}, Dice={val_dice[-1]:.4f}") # mean dice score across all 6 classes 

    # save model weights and plot metrics 
    torch.save(model.state_dict(), "improved_unet3d.pth")
    plot_metrics(train_loss, val_loss, val_dice, val_dice_per_class)

if __name__ == "__main__":
    train_model()
    