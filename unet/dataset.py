import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import random


class SegmentationDataset(Dataset):
    """Generic image segmentation dataset.

    Expects:
        root/images/*.png  (or jpg)
        root/masks/*.png
    """

    def __init__(self, root, image_size=(256, 256), augment=False):
        self.image_dir = os.path.join(root, "images")
        self.mask_dir = os.path.join(root, "masks")
        self.image_size = image_size
        self.augment = augment

        self.ids = sorted([
            f for f in os.listdir(self.image_dir)
            if f.endswith((".png", ".jpg", ".jpeg"))
        ])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        name = self.ids[idx]
        image = Image.open(os.path.join(self.image_dir, name)).convert("L")
        mask = Image.open(os.path.join(self.mask_dir, name)).convert("L")

        image = image.resize(self.image_size, Image.BILINEAR)
        mask = mask.resize(self.image_size, Image.NEAREST)

        if self.augment:
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            if random.random() > 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)

        image = TF.to_tensor(image)
        mask = torch.from_numpy(np.array(mask)).float().unsqueeze(0) / 255.0
        mask = (mask > 0.5).float()

        return image, mask


class SyntheticDataset(Dataset):
    """Synthetic dataset for quick testing without real data."""

    def __init__(self, size=200, image_size=(256, 256)):
        self.size = size
        self.h, self.w = image_size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        image = torch.rand(1, self.h, self.w)
        mask = torch.zeros(1, self.h, self.w)
        cx = torch.randint(64, self.w - 64, (1,)).item()
        cy = torch.randint(64, self.h - 64, (1,)).item()
        r = torch.randint(20, 60, (1,)).item()
        y, x = torch.meshgrid(torch.arange(self.h), torch.arange(self.w), indexing="ij")
        mask[0] = ((x - cx) ** 2 + (y - cy) ** 2 < r ** 2).float()
        return image, mask
