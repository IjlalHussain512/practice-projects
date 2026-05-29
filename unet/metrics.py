import torch


def dice_score(preds, targets, threshold=0.5, smooth=1e-6):
    preds = (torch.sigmoid(preds) > threshold).float()
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    return ((2 * intersection + smooth) / (union + smooth)).mean()


def iou_score(preds, targets, threshold=0.5, smooth=1e-6):
    preds = (torch.sigmoid(preds) > threshold).float()
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) - intersection
    return ((intersection + smooth) / (union + smooth)).mean()
