"""
utils.py

Dice and Plotting
"""

import torch
import matplotlib.pyplot as plt

def dice_score(pred, target, eps=1e-6):
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    return (2. * intersection) / (pred.sum() + target.sum() + eps)

def plot_metrics(losses, dices):
    # Convert tensors to CPU and NumPy
    losses = [l.item() if torch.is_tensor(l) else l for l in losses]
    dices = [d.item() if torch.is_tensor(d) else d for d in dices]

    plt.figure()
    plt.plot(losses, label='Loss')
    plt.plot(dices, label='Dice')
    plt.xlabel('Epoch')
    plt.ylabel('Metric')
    plt.legend()
    plt.title('Training Metrics')
    plt.savefig("metrics.png")
    plt.close()