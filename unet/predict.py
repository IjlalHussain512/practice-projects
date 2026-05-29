import argparse
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from model import UNet


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True, help="Path to input image")
    p.add_argument("--checkpoint", default="unet_best.pth")
    p.add_argument("--img-size", type=int, default=256)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--output", default="prediction.png")
    return p.parse_args()


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    size = (args.img_size, args.img_size)
    image = Image.open(args.image).convert("L").resize(size, Image.BILINEAR)
    x = torch.from_numpy(np.array(image)).float().unsqueeze(0).unsqueeze(0) / 255.0
    x = x.to(device)

    with torch.no_grad():
        pred = torch.sigmoid(model(x)).squeeze().cpu().numpy()

    mask = (pred > args.threshold).astype(np.uint8) * 255

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(np.array(image), cmap="gray"); axes[0].set_title("Input")
    axes[1].imshow(pred, cmap="hot");             axes[1].set_title("Probability")
    axes[2].imshow(mask, cmap="gray");            axes[2].set_title("Prediction")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Saved prediction to {args.output}")


if __name__ == "__main__":
    main()
