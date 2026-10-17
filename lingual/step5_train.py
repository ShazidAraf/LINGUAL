"""STEP 5 - train on the labelled pool, and evaluate on the target test set.

Loss on pool pixels only (255 = ignored), taken from Acquisition_Function/<method>/execute.py:
  RIPU    : CE + 0.1 * negative-learning loss (Xie et al., CVPR 2022)
  Entropy : CE
"""
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from lingual.metrics import eval_metrics, effective_spacing
from lingual.step1_selection import acquisition_loss


def train_epoch(model, loader, optimizer, cfg):
    model.train()
    total, n = 0.0, 0
    for images, _, _, pool in loader:  # GT is not used for training
        images, pool = images.to(cfg.device), pool.long().to(cfg.device)
        optimizer.zero_grad()
        loss = acquisition_loss(model(images), pool, cfg)
        loss.backward()
        optimizer.step()
        total, n = total + loss.item(), n + 1
    return total / max(n, 1)


@torch.no_grad()
def evaluate(model, dataset, cfg, save_file):
    """Per-class DSC / RAVD / HD95 / ASSD on the target test set, appended to save_file."""
    model.eval()
    results = []
    distance_mode = getattr(cfg, "distance_mode", "voxel")
    slice_stores = {c: {} for c in range(1, cfg.num_classes)}

    for i in tqdm(range(len(dataset)), desc="eval"):
        img, label, name = dataset[i]
        sp = effective_spacing(cfg, dataset, name)
        if img.dim() == 3:  # single 2D image
            img, label = img.unsqueeze(0), label.unsqueeze(0)
        preds = torch.argmax(model(img.to(cfg.device)), dim=1)
        pred_oh = F.one_hot(preds, cfg.num_classes).permute(0, 3, 1, 2).cpu().numpy()
        gt_oh = F.one_hot(label.long().to(cfg.device), cfg.num_classes).permute(0, 3, 1, 2).cpu().numpy()

        per_class = []
        for c in range(1, cfg.num_classes):
            p, g = pred_oh[:, c], gt_oh[:, c]
            if p.shape[0] == 1:
                p, g = p[0], g[0]
            per_class.append(eval_metrics(g, p, distance_mode=distance_mode, slice_store=slice_stores[c], spacing=sp))
        results.append(per_class)

    r = np.array(results)
    dsc, ravd = np.mean(r[:, :, 0], axis=0), np.mean(r[:, :, 1], axis=0)
    hd95, assd = np.nanmean(r[:, :, 4], axis=0), np.nanmean(r[:, :, 5], axis=0)
    if distance_mode == "pixel" and any(slice_stores[c] for c in slice_stores):
        hd95 = np.array([np.nanmean(slice_stores[c]["hd95"]) for c in range(1, cfg.num_classes)])
        assd = np.array([np.nanmean(slice_stores[c]["assd"]) for c in range(1, cfg.num_classes)])

    summary = (f"DSC {np.mean(dsc):.4f} | RAVD {np.mean(ravd):.4f} | "
               f"HD95 {np.nanmean(hd95):.4f} | ASSD {np.nanmean(assd):.4f}")
    print(summary)
    with open(save_file, "a") as f:
        f.write(f"[distance_mode: {distance_mode} | units: {getattr(cfg, 'distance_units', 'px')}]\n")
        for metric, vals in (("DSC", dsc), ("RAVD", ravd), ("HD95", hd95), ("ASSD", assd)):
            f.write(f"Per-class {metric} mean:\n")
            for c, v in enumerate(vals, start=1):
                f.write(f"Class {c}: {v:.4f}\n")
            f.write(f"Overall Avg {metric}: {np.nanmean(vals) if metric in ('HD95', 'ASSD') else np.mean(vals):.4f}\n")
        f.write("\n\n\n")
    model.train()
    return float(np.mean(dsc))
