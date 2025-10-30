"""
    File name: utils.py
    Author: Louisa Wu
    Date created: 13/10/2025
    Date last modified: 31/10/2025
    Python Version: 3.9.23

    Description: 
        Utility functions for training, evaluation and visualisation of the Improved 3D UNet model 
        on the HipMRI Prostate dataset.
"""

import torch
import matplotlib.pyplot as plt

def dice_score(pred, target, num_classes=6, eps=1e-6):
    """ Computes the Dice similarity scores for multi-class segmentation """
    pred = torch.argmax(pred, dim=1)  # convert probabilities to class indices 
    dice_scores = []

    for c in range(num_classes):
        pred_c = (pred == c).float()
        target_c = (target == c).float()
        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()
        dice = (2. * intersection + eps) / (union + eps)
        dice_scores.append(dice.item())

    return dice_scores

def plot_metrics(train_loss, val_loss, val_dice, val_dice_per_class):
    """ Plots training and validation metrics, including loss curves and Dice scores """
    # training and validation loss 
    plt.figure()
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig("loss_plot.png")
    plt.close()

    # mean Dice score 
    plt.figure()
    plt.plot(val_dice, label='Mean Dice')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Score')
    plt.title('Validation Mean Dice')
    plt.legend()
    plt.savefig("mean_dice_plot.png")
    plt.close()

    # Dice scores per class 
    plt.figure()
    for c in range(6):
        plt.plot(val_dice_per_class[c], label=f'Class {c}')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Score')
    plt.title('Per-Class Dice Scores')
    plt.legend()
    plt.savefig("per_class_dice_plot.png")
    plt.close()
