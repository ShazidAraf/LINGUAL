"""Geometry helpers for the modules (from GraCo-main/FUNCTIONS/highlight_region.py)."""
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial.distance import cdist


def border_band(mask, r1, r2):
    """Pixels whose distance to the foreground of `mask` is in (r1, r2]."""
    if not np.any(mask):
        return np.zeros_like(mask, dtype=np.float32)
    d = distance_transform_edt(1 - mask)
    return ((d > r1) & (d <= r2)).astype(np.float32)


def get_border_regions(mask, r_inner, r_outer):
    """Thin bands just inside and just outside the mask boundary."""
    return border_band(1 - mask, r_inner, r_outer), border_band(mask, r_inner, r_outer)


def sample_uniformly_spaced_points(mask, n):
    """Farthest-point sampling of n foreground pixels, (row, col)."""
    coords = np.argwhere(mask > 0)
    if len(coords) < n:
        raise ValueError(f"Not enough foreground points to sample {n} points.")
    sampled = [np.random.randint(0, len(coords))]
    for _ in range(1, n):
        sampled.append(np.argmax(np.min(cdist(coords[sampled], coords), axis=0)))
    return coords[sampled]


# Shift applied to the centre (row, col) before cutting the half-plane: move AGAINST the direction
_SHIFT = {"right": (0, -1), "left": (0, 1), "top": (1, 0), "bottom": (-1, 0),
          "top-right": (1, -1), "top-left": (1, 1), "bottom-right": (-1, -1), "bottom-left": (-1, 1)}

# Each direction keeps a 180-degree sector (angles in image coords, 0 = right, 90 = down)
_SECTOR = {"right": (270, 90), "bottom-right": (315, 135), "bottom": (0, 180), "bottom-left": (45, 225),
           "left": (90, 270), "top-left": (135, 315), "top": (180, 360), "top-right": (225, 45)}


def weight_highlight_region(mask, center, direction, adjust_len=40):
    """Binary map of the half-plane facing `direction` (list of names), seen from `center` (row, col)."""
    h, w = mask.shape
    dy, dx = _SHIFT.get(direction[0], (0, 0))
    cy = np.clip(center[0] + dy * adjust_len, 0, h - 1)
    cx = np.clip(center[1] + dx * adjust_len, 0, w - 1)

    Y, X = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    angles = (np.degrees(np.arctan2(Y - cy, X - cx)) + 360) % 360
    out = np.zeros_like(mask, dtype=np.float32)
    for d in direction:
        if d not in _SECTOR:
            raise ValueError(f"Unsupported direction: {d}")
        lo, hi = _SECTOR[d]
        out[((angles >= lo) & (angles <= hi)) if lo < hi else ((angles >= lo) | (angles <= hi))] = 1
    return out
