"""STEP 1 - the model finds uncertain patches.

Each image gets `cfg.n_budget_box` square boxes of side `cfg.patch_size`,
returned as [x1, y1, x2, y2]. Pixels already labelled (pool mask != 255) are never re-selected.

Acquisition functions in ../Acquisition_Function/{RIPU,Entropy}/
  RIPU    : region impurity x region-averaged entropy (Xie et al., CVPR 2022, "RIPU")
  Entropy : region-averaged prediction entropy
If an image gets fewer boxes than the budget, `top_up_boxes` adds boxes at the highest
pixel-entropy peaks.
"""
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # LINGUAL_clean/
from Acquisition_Function.RIPU.execute import get_ROI_RIPU, loss_RIPU
from Acquisition_Function.Entropy.execute import get_ROI_Entropy, loss_Entropy

ACQUISITION = {"RIPU": (get_ROI_RIPU, loss_RIPU), "Entropy": (get_ROI_Entropy, loss_Entropy)}


def select_boxes(model, images, pool_masks, cfg):
    """images: [B,3,H,W], pool_masks: [B,H,W]. Returns per image a list of [x1, y1, x2, y2]."""
    get_roi, _ = ACQUISITION[cfg.method]
    return get_roi(model, images, pool_masks.to(images.device), cfg)


def acquisition_loss(logits, pool_mask, cfg):
    """Training loss of the chosen method (RIPU: CE + 0.1 negative learning, Entropy: CE)."""
    _, loss_fn = ACQUISITION[cfg.method]
    return loss_fn(None, logits, pool_mask, cfg)


@torch.no_grad()
def top_up_boxes(model, image, pool_mask, boxes, n_box, cfg):
    """Add n_box boxes at pixel-entropy peaks, avoiding labelled pixels and existing boxes."""
    model.eval()
    p = F.softmax(model(image.unsqueeze(0)), dim=1)
    entropy = -torch.sum(p * torch.log(p + 1e-8), dim=1)[0].cpu().numpy()
    score = (entropy - entropy.min()) / (entropy.max() - entropy.min() + 1e-8)
    score[pool_mask.cpu().numpy() != 255] = -np.inf
    for x1, y1, x2, y2 in boxes:
        score[y1:y2, x1:x2] = -np.inf

    h, w = score.shape
    r_active, r_mask = cfg.patch_size // 2, cfg.patch_size
    new = []
    for _ in range(n_box):
        if np.max(score) == -np.inf:
            break
        cy, cx = np.unravel_index(np.argmax(score), score.shape)
        new.append([max(0, cx - r_active), max(0, cy - r_active + 1), min(w, cx + r_active + 1), min(h, cy + r_active)])
        score[max(0, cy - r_mask):min(h, cy + r_mask + 1), max(0, cx - r_mask):min(w, cx + r_mask + 1)] = -np.inf
    model.train()
    return boxes + new
