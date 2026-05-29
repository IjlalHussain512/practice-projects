import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from model import UNet
from dataset import SyntheticDataset, SegmentationDataset
from metrics import dice_score


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=None, help="Path to dataset root (images/ + masks/). Leave empty to use synthetic data.")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--img-size", type=int, default=256)
    p.add_argument("--checkpoint", default="unet_best.pth")
    return p.parse_args()


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        preds = model(images)
        loss = criterion(preds, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, total_dice = 0, 0
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        preds = model(images)
        total_loss += criterion(preds, masks).item()
        total_dice += dice_score(preds, masks).item()
    return total_loss / len(loader), total_dice / len(loader)


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    size = (args.img_size, args.img_size)
    if args.data:
        full_ds = SegmentationDataset(args.data, image_size=size, augment=True)
        n_val = max(1, int(len(full_ds) * 0.1))
        train_ds, val_ds = random_split(full_ds, [len(full_ds) - n_val, n_val])
    else:
        print("No --data provided. Using synthetic dataset.")
        train_ds = SyntheticDataset(size=160, image_size=size)
        val_ds = SyntheticDataset(size=40, image_size=size)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, num_workers=2, pin_memory=True)

    model = UNet(in_channels=1, out_channels=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    criterion = nn.BCEWithLogitsLoss()

    best_dice = 0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_dice = eval_epoch(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        print(f"Epoch {epoch:03d}/{args.epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_dice={val_dice:.4f}")

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), args.checkpoint)
            print(f"  -> Saved checkpoint (dice={best_dice:.4f})")

    print(f"\nTraining complete. Best Dice: {best_dice:.4f}")


if __name__ == "__main__":
    main()
