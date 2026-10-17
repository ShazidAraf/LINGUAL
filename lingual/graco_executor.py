"""EXPAND / SHRINK executor driven by GraCo (Zhao et al., CVPR 2024, granularity-controllable interactive seg).

Clean copy of GraCo-main/FUNCTIONS/Expand_Shrink_v8.py (that file is never edited).
Algorithm and constants unchanged; only debug plotting removed and the predictor loaded lazily.

Idea per iteration:
  1. take thin bands just inside / just outside the current boundary, inside the box, on the side
     facing the requested direction;
  2. click points on each band, run GraCo at granularity `gra` -> map_in, map_out;
  3. eta = IoU(map_in, map_out); expand/shrink the mask with the maps;
  4. stop on a local minimum / sharp drop of eta; return the mask with the lowest eta.
"""
import copy
import os
import sys

import cv2
import numpy as np
import torch
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects, skeletonize

from lingual.regions import get_border_regions, sample_uniformly_spaced_points, weight_highlight_region

# GraCo code (isegm) is copied in third_party/graco; only the weights live outside this folder
GRACO_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "third_party", "graco")
GRACO_WEIGHTS = os.environ.get("GRACO_WEIGHTS", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weights"))
sys.path.append(GRACO_ROOT)

gra_shrink_factor = 0.7
sampled_point_th = 0.25
point_percent = 0.2
min_num_points = 3
iou_min = 0.05
graco_foreground_th = 0.7
r_in, r_out = 3, 5

_predictor = None
init_p = 100


def get_predictor():
    global _predictor
    if _predictor is None:
        from isegm.inference import utils
        from isegm.inference.predictors import get_predictor as _get
        model = utils.load_is_model(os.path.join(GRACO_WEIGHTS, "simple_click/sbd_vit_large.pth"), "cuda:0", 0,
                                    os.path.join(GRACO_WEIGHTS, "graco/GraCo_large_lora.pth"))
        _predictor = _get(model, "NoBRS", "cuda:0", prob_thresh=0.5)
    return _predictor


def get_iou(a, b):
    """Same as isegm.inference.utils.get_iou (nan when both maps are empty)."""
    a, b = a == 1, b > 0
    return np.float64(np.logical_and(a, b).sum()) / np.logical_or(a, b).sum()


# ------------------- mask helpers -------------------
def keep_largest_cc(mask):
    lab = label(mask, connectivity=2)
    regions = regionprops(lab)
    if not regions:
        return np.zeros_like(mask)
    return (lab == max(regions, key=lambda r: r.area).label).astype(np.uint8)


def thin(mask):
    return skeletonize(mask > 0).astype(np.uint8)


def extract_boundary(mask):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    boundary = np.zeros_like(mask, dtype=np.uint8)
    cv2.drawContours(boundary, contours, -1, 1, 1)
    return boundary


def closest_point_in_box(binary, x, y, w, h):
    """Foreground pixel of `binary` inside the box closest to the box centre, (row, col)."""
    H, W = binary.shape
    x1, y1, x2, y2 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    pts = np.argwhere(binary[y1:y2, x1:x2] > 0)
    if len(pts) == 0:
        return None
    pts = pts + np.array([y1, x1])
    return tuple(pts[np.argmin(np.linalg.norm(pts - np.array([y + h / 2, x + w / 2]), axis=1))])


def clean_binary_mask(mask, min_size=100, k=5):
    mask = remove_small_objects((mask > 0), min_size=min_size).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel), cv2.MORPH_OPEN, kernel)


def update_expand(pred, map_in, map_out):
    return np.logical_or(pred, np.logical_or(map_in, np.logical_and(map_in, map_out)))


def update_shrink(pred, map_in, map_out):
    return np.logical_and(pred == 1, np.logical_and(map_in, map_out) == 0)


# ------------------- GraCo maps -------------------
def graco_union(img, mask, coords, gra):
    """Union of GraCo masks, one positive click per point, at granularity gra."""
    from isegm.inference.clicker import Clicker
    predictor = get_predictor()
    out = None
    for c in coords:
        clicker = Clicker(gt_mask=mask)
        clicker.set_click_manually(np.array([c]), is_positive=1)
        predictor.set_input_image(img)
        predictor.prev_prediction = torch.zeros_like(predictor.original_image[:, :1, :, :])
        m = predictor.get_prediction(clicker, gra=gra) > graco_foreground_th
        out = m if out is None else np.logical_or(out, m)
    return out


def graco_maps(img, pred, box, direction, gra, iteration):
    """Returns (map_in, map_out, stop_flag, box_mask). box = [top_y, top_x, h, w]."""
    global init_p
    ty, tx, h, w = box
    box_mask = np.zeros_like(img[:, :, 0])
    box_mask[ty:ty + h, tx:tx + w] = 1

    inner, outer = get_border_regions(pred, r_in, r_out)
    inner = keep_largest_cc(thin(inner) * box_mask)
    outer = keep_largest_cc(thin(outer) * box_mask)

    anchor = closest_point_in_box(extract_boundary(pred), tx, ty, w, h)
    w_dir = weight_highlight_region(pred, np.array(anchor), direction, adjust_len=40)
    mask_in, mask_out = inner * w_dir, outer * w_dir

    n_points = int(point_percent * min(mask_in.sum(), mask_out.sum()))
    if iteration == 1:
        init_p = n_points
    if n_points < min_num_points or n_points < sampled_point_th * init_p or n_points > init_p / sampled_point_th:
        return None, None, 1, box_mask

    map_in = graco_union(img, pred, sample_uniformly_spaced_points(mask_in, n_points), gra)
    map_out = graco_union(img, pred, sample_uniformly_spaced_points(mask_out, n_points), gra)
    return map_in, map_out, 0, box_mask


# ------------------- main entry -------------------
def MODIFY(img, pred, box, direction, mode="expand"):
    """img: HxWx3 uint8, pred: HxW binary, box: [top_y, top_x, h, w], direction: list of names."""
    if box[2] < 45 or box[3] < 45:
        gra, max_iter = 0.1, 10
    elif box[2] < 70 or box[3] < 70:
        gra, max_iter = 0.1, 15
    else:
        gra, max_iter = 0.2, 20

    all_iou, all_preds, iou_vals = [], [], []
    best_iou, best_pred = 1.0, None
    iteration = 1

    while True:
        map_in, map_out, stop, box_mask = graco_maps(img, pred, box, direction, gra, iteration)

        if stop == 1 or iteration > max_iter:
            if best_pred is None:
                best_pred = pred if not all_iou else all_preds[int(np.argmin(all_iou))]
            break

        iou = get_iou(map_in, map_out)
        all_iou.append(iou)

        # sharp eta drop (>30%) = arrival at the boundary: keep the previous state and stop
        if len(all_iou) >= 2 and (all_iou[-2] - all_iou[-1]) / (all_iou[-2] + 1e-8) > 0.3:
            if best_pred is None:
                best_pred = all_preds[-2] if len(all_preds) >= 2 else pred
            break

        map_in, map_out = map_in * box_mask, map_out * box_mask
        pred = update_expand(pred, map_in, map_out) if mode == "expand" else update_shrink(pred, map_in, map_out)
        pred = clean_binary_mask(pred, min_size=100, k=5)
        all_preds.append(pred)
        iou_vals.append(iou)

        # local minimum of eta: remember it, restart from before the valley with finer granularity
        if len(iou_vals) >= 3 and iou_vals[-3] > iou_vals[-2] < iou_vals[-1]:
            if best_iou > iou_vals[-2]:
                best_pred, best_iou = all_preds[-2], all_iou[-2]
                if best_iou < iou_min:
                    break
            pred = all_preds[-3]
            gra *= gra_shrink_factor
            iou_vals = []

        iteration += 1

    return best_pred
