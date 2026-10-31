"""Pérdida y métricas binarias: solo píxeles válidos (sin NoData)."""
import torch
from torch.nn import functional as F


def masked_bce(logits, labels, valid):
    loss_per_pixel = F.binary_cross_entropy_with_logits(
        logits, labels, reduction="none"
    )
    return (loss_per_pixel * valid).sum() / valid.sum().clamp_min(1.0)



def masked_dice(logits, labels, valid, smooth=1.0):
    """Soft Dice loss sobre agua (clase 1); ignora los pixeles NoData.

    Calcula Dice por chip y promedia solo los chips con pixeles validos.
    """
    probs = torch.sigmoid(logits) * valid
    targets = labels * valid
    dims = (1, 2, 3)
    intersection = (probs * targets).sum(dim=dims)
    denom = probs.sum(dim=dims) + targets.sum(dim=dims)
    dice_loss = 1.0 - (2.0 * intersection + smooth) / (denom + smooth)
    has_valid = valid.sum(dim=dims) > 0
    if not has_valid.any():
        return logits.sum() * 0.0
    return dice_loss[has_valid].mean()


def segmentation_loss(logits, labels, valid, name="bce"):
    """M1: BCE. M2: 0.5 * BCE + 0.5 * Soft Dice."""
    bce = masked_bce(logits, labels, valid)
    if name == "bce":
        return bce
    if name == "bce_dice":
        return 0.5 * bce + 0.5 * masked_dice(logits, labels, valid)
    raise ValueError(f"Funcion de perdida no admitida: {name}")


class WaterMetrics:
    """IoU global + media IoU por chip, F1, precisión, recall y confusión."""

    def __init__(self):
        self.tp = self.fp = self.tn = self.fn = 0
        self.chip_ious = []
        self.n_chips = 0
        self.n_valid = 0

    @torch.no_grad()
    def update(self, logits, labels, valid):
        p = torch.sigmoid(logits) >= 0.5
        y = labels >= 0.5
        v = valid > 0.5
        for i in range(p.shape[0]):
            pi, yi, vi = p[i], y[i], v[i]
            tp = int((pi & yi & vi).sum().item())
            fp = int((pi & ~yi & vi).sum().item())
            fn = int((~pi & yi & vi).sum().item())
            tn = int((~pi & ~yi & vi).sum().item())
            self.tp += tp
            self.fp += fp
            self.fn += fn
            self.tn += tn
            self.n_chips += 1
            self.n_valid += tp + fp + fn + tn
            if tp + fp + fn:
                self.chip_ious.append(tp / (tp + fp + fn))

    def summary(self):
        tp, fp, fn = self.tp, self.fp, self.fn
        return {
            "iou_global": tp / (tp + fp + fn) if tp + fp + fn else None,
            "iou_media_chips": (
                sum(self.chip_ious) / len(self.chip_ious) if self.chip_ious else None
            ),
            "f1_agua": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
            "precision_agua": tp / (tp + fp) if tp + fp else None,
            "recall_agua": tp / (tp + fn) if tp + fn else None,
            "tp": tp, "fp": fp, "tn": self.tn, "fn": fn,
            "chips": self.n_chips,
            "chips_iou_definido": len(self.chip_ious),
            "pixeles_validos": self.n_valid,
        }
