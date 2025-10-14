import torch
from modules import UNet2D
from dataset import Prostate2DDataset
import matplotlib.pyplot as plt
from utils import dice_score

def predict():
    model = UNet2D().cuda()
    model.load_state_dict(torch.load("unet2d.pth"))
    model.eval()

    ds = Prostate2DDataset("/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_test",
                           "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_seg_test")
    x, y = ds[0]
    with torch.no_grad():
        pred = model(x.unsqueeze(0).cuda()).cpu().squeeze().numpy()

    # Threshold prediction and convert to tensor
    pred_bin = torch.tensor(pred > 0.5, dtype=torch.float32)

    # Compute Dice score
    dice = dice_score(pred_bin, y.cpu())
    print(f"Dice score on test sample: {dice:.4f}", flush=True)

    plt.subplot(1, 3, 1); plt.imshow(x.squeeze(), cmap='gray'); plt.title("Input")
    plt.subplot(1, 3, 2); plt.imshow(y.squeeze(), cmap='gray'); plt.title("Ground Truth")
    plt.subplot(1, 3, 3); plt.imshow(pred > 0.5, cmap='gray'); plt.title("Prediction")
    plt.tight_layout()
    plt.savefig("prediction_result.png")
    print("Saved prediction_result.png", flush=True)

if __name__ == "__main__":
    predict()
