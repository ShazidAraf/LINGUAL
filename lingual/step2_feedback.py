"""STEP 2 - simulated human language feedback for one box.

The "human" looks at the ground truth inside the box and says what is wrong:
    "Entire region Foreground" / "Entire region Background" / "No change"
    "<Expand|Shrink> the border to <direction>, Remove the fragments, Fill up the holes, Smooth the Border"
Ground truth is used ONLY here, to simulate the annotator. The executors in step 4 never see it.
"""
import cv2
import numpy as np

DIRECTIONS = [(0, "right"), (45, "top-right"), (90, "top"), (135, "top-left"),
              (180, "left"), (225, "bottom-left"), (270, "bottom"), (315, "bottom-right")]


def dominant_class(ref, pred, box, bg_label):
    """Most frequent non-background GT class in the box -> binary GT / pred masks for that class."""
    x1, y1, x2, y2 = box
    labels, counts = np.unique(ref[y1:y2, x1:x2], return_counts=True)
    counts = {l: c for l, c in zip(labels, counts) if l != bg_label}
    if not counts:
        return np.zeros_like(ref, np.uint8), np.zeros_like(pred, np.uint8), bg_label
    label = max(counts, key=counts.get)
    return (ref == label).astype(np.uint8), (pred == label).astype(np.uint8), label


def largest_component(mask):
    mask = (mask > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = np.zeros_like(mask)
    if n > 1:
        out[labels == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])] = 1
    return out


def operation_and_direction(ref, pred, box):
    """Rule-based annotator. Returns (operation, direction); direction is -1 for whole-box ops."""
    H, W = ref.shape
    x1, y1, x2, y2 = box
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)

    pred = largest_component(pred)
    ref_p, pred_p = ref[y1:y2, x1:x2], pred[y1:y2, x1:x2]

    if ref_p.sum() == 0 and pred_p.sum() == 0:
        return "Background", -1
    fg = ref_p.sum() / ref_p.size
    if fg < 0.10:
        return "Background", -1
    if fg > 0.90:
        return "Foreground", -1
    dsc = 2 * (ref_p * pred_p).sum() / (ref_p.sum() + pred_p.sum() + 1e-6)
    if dsc > 0.90:
        return "No change", -1

    FN = (ref_p == 1) & (pred_p == 0)
    FP = (pred_p == 1) & (ref_p == 0)
    operation, diff = ("Expand", FN) if FN.sum() >= FP.sum() else ("Shrink", FP)

    if pred_p.sum() > 0:
        if diff.sum() / (pred_p.sum() + 1e-6) < 0.05:
            return "No change", -1
    else:  # nothing predicted in the box
        if fg < 0.30:
            return "Background", -1
        if fg > 0.70:
            return "Foreground", -1
        return "Ignore", -1

    # Direction = compass angle from the predicted-mask centroid to the error centroid
    dy_, dx_ = np.where(diff)
    py_, px_ = np.where(pred_p)
    angle = np.degrees(np.arctan2(-(dy_.mean() - py_.mean()), dx_.mean() - px_.mean())) % 360
    direction = min(DIRECTIONS, key=lambda d: abs((angle - d[0] + 180) % 360 - 180))[1]
    return operation, direction


def has_fragments(pred, ref, threshold=200):
    """Small (< threshold px) predicted components that lie outside the GT."""
    pred = (pred > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(pred, connectivity=8)
    small = np.isin(labels, [l for l in range(1, n) if stats[l, cv2.CC_STAT_AREA] < threshold])
    return bool(np.any(small & (ref == 0)))


def has_holes(pred, max_area=200):
    """Background components of the prediction with area <= max_area."""
    n, _, stats, _ = cv2.connectedComponentsWithStats(1 - (pred > 0).astype(np.uint8), connectivity=8)
    return bool(np.any(stats[1:, cv2.CC_STAT_AREA] <= max_area))


def simulate_feedback(ref_total, pred_total, box, cfg):
    """Returns (operation, direction, keywords, ref_bin, pred_bin, dominant_label)."""
    ref, pred, label = dominant_class(ref_total, pred_total, box, cfg.bg_label)
    if label == cfg.bg_label:
        return "Background", -1, ["Entire region Background"], ref, pred, label

    operation, direction = operation_and_direction(ref, pred, box)
    if direction == -1:
        keywords = {"No change": ["No change"],
                    "Foreground": ["Entire region Foreground"],
                    "Background": ["Entire region Background"]}.get(operation, [])
        return operation, direction, keywords, ref, pred, label

    x1, y1, x2, y2 = box
    keywords = [f"{operation} the border to {direction}"]
    if has_fragments(pred[y1:y2 + 1, x1:x2 + 1], ref[y1:y2 + 1, x1:x2 + 1]):
        keywords.append(" Remove the fragments ")
    if has_holes(pred[y1:y2 + 1, x1:x2 + 1]):
        keywords.append(" Fill up the holes ")
    keywords.append(" Smooth the Border")
    return operation, direction, keywords, ref, pred, label
