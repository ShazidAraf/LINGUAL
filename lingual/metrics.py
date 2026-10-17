# -*- coding: utf-8 -*-
"""
Created on 09/07/2019

@author: Ali Emre Kavur
"""
# import pydicom
import os
import numpy as np
import glob, pdb
import cv2
# import SimpleITK as sitk
from scipy import ndimage
from sklearn.neighbors import KDTree
import torch
import numpy as np
from scipy.spatial.distance import cdist
from scipy.ndimage import binary_erosion
import numpy as np
from sklearn.metrics import roc_auc_score

from lingual.surface_dice import compute_surface_distances, compute_surface_dice_at_tolerance



########################


def DICE(Vref,Vseg):
    dice=2*(torch.mul(Vref,Vseg)).sum()/(Vref.sum() + Vseg.sum()+1e-08)
    return dice

def RAVD(Vref,Vseg):
    ravd=(abs(Vref.sum() - Vseg.sum())/(Vref.sum())+1e-10)
    return ravd

def compute_dice_score(gt, pred, epsilon=1e-8):

    mask1, mask2 = gt.astype(np.bool_), pred.astype(np.bool_)
    intersection = np.sum(mask1 & mask2)
    return (2.0 * intersection) / (np.sum(mask1) + np.sum(mask2) + epsilon)

def compute_RAVD(gt, pred, epsilon=1e-8):
    mask1, mask2 = gt.astype(np.bool_), pred.astype(np.bool_)
    ravd = (abs(np.sum(mask1) - np.sum(mask2)) / (np.sum(mask1) + epsilon))
    return ravd


def compute_auc(y_true, y_pred):
    # Flatten the maps to 1D arrays
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    # Check if both inputs are same shape
    assert y_true_flat.shape == y_pred_flat.shape, "Shape mismatch between ground truth and prediction."

    if np.sum(y_true_flat)==0 or np.sum(y_pred_flat)==0:
        return np.nan

    # Compute AUC
    auc = roc_auc_score(y_true_flat, y_pred_flat)
    return auc


def compute_nsd(gt_mask, pred_mask, tolerance=2.0, spacing=(1.0, 1.0)):
    """
    Compute the Normalized Surface Dice (NSD) between two binary masks.
    
    Parameters:
    - gt_mask (ndarray): Ground truth binary mask (2D or 3D), values in {0, 1}
    - pred_mask (ndarray): Predicted binary mask, same shape and type as gt_mask
    - tolerance (float): Distance tolerance in pixels or mm for NSD
    - spacing (tuple): Voxel spacing (for 2D masks, use (y_spacing, x_spacing))

    Returns:
    - nsd (float): Normalized Surface Dice score
    """

    # Ensure binary masks are uint8
    gt_mask = (gt_mask > 0).astype(np.uint8)
    pred_mask = (pred_mask > 0).astype(np.uint8)

    # Add singleton dimension if needed (to simulate 3D for SurfaceDice functions)
    if gt_mask.ndim == 2:
        gt_mask = gt_mask[..., None]
        pred_mask = pred_mask[..., None]

    # Pad spacing with 1.0 per missing axis so 3D volumes work too
    spacing = tuple(spacing) + (1.0,) * (gt_mask.ndim - len(spacing))

    # Compute surface distances
    surface_distances = compute_surface_distances(gt_mask, pred_mask, spacing)

    # Compute NSD
    nsd = compute_surface_dice_at_tolerance(surface_distances, tolerance)

    return nsd


def surface_distances_oneway(mask_a, mask_b, spacing=None):
    """
    Distances from every border pixel of mask_a to the nearest border pixel
    of mask_b (medpy-style, using an exact euclidean distance transform).
    Works for both 2D images and 3D volumes; spacing defaults to 1.0 per axis.
    """
    mask_a = mask_a.astype(bool)
    mask_b = mask_b.astype(bool)

    if spacing is None:
        spacing = (1.0,) * mask_a.ndim

    struct = ndimage.generate_binary_structure(mask_a.ndim, 1)
    border_a = mask_a ^ ndimage.binary_erosion(mask_a, structure=struct)
    border_b = mask_b ^ ndimage.binary_erosion(mask_b, structure=struct)

    dt = ndimage.distance_transform_edt(~border_b, sampling=spacing)
    return dt[border_a]


def compute_hd95(gt, pred, spacing=None):
    """
    95th percentile symmetric Hausdorff Distance (in pixels/voxels unless
    spacing is given). Handles 2D images and 3D volumes.
    Returns nan if either mask is empty.
    """
    gt = gt.astype(bool)
    pred = pred.astype(bool)

    if spacing is None:
        spacing = (1.0,) * gt.ndim

    if not gt.any() or not pred.any():
        return np.nan

    d_pred_to_gt = surface_distances_oneway(pred, gt, spacing)
    d_gt_to_pred = surface_distances_oneway(gt, pred, spacing)
    return np.percentile(np.hstack((d_pred_to_gt, d_gt_to_pred)), 95)


def compute_assd(gt, pred, spacing=None):
    """
    Average Symmetric Surface Distance (in pixels/voxels unless spacing is
    given). Handles 2D images and 3D volumes.
    Returns nan if either mask is empty.
    """
    gt = gt.astype(bool)
    pred = pred.astype(bool)

    if spacing is None:
        spacing = (1.0,) * gt.ndim

    if not gt.any() or not pred.any():
        return np.nan

    d_pred_to_gt = surface_distances_oneway(pred, gt, spacing)
    d_gt_to_pred = surface_distances_oneway(gt, pred, spacing)
    return np.mean((d_pred_to_gt.mean(), d_gt_to_pred.mean()))


def eval_metrics(gt,pred, distance_mode="voxel", slice_store=None, spacing=None):
    """
    distance_mode controls HD95/ASSD for 3D stack inputs (CHAOS):
        "voxel" (default): volumetric 3D surface distances per stack (voxel units)
        "pixel"          : per-2D-slice distances averaged (pixel units, comparable
                           with the per-image 2D datasets). If slice_store (a dict)
                           is given, every per-slice value is also appended to it
                           (keys "hd95"/"assd") so the caller can pool ALL slices of
                           ALL stacks into one flat average instead of stack-by-stack.
    2D inputs are unaffected by this flag.
    spacing: None -> pixel/voxel units. Otherwise physical spacing in mm per axis on the
             evaluation grid: (sy, sx) for 2D, (sz, sy, sx) for 3D stacks; HD95/ASSD are then in mm.
    """
    # pdb.set_trace()
    
    pred = (pred > 0).astype(bool)
    gt = (gt > 0).astype(bool)

    if np.sum(gt)==0 and np.sum(pred)==0:
        return np.nan,np.nan,np.nan,np.nan,np.nan,np.nan

    dsc = compute_dice_score(gt,pred)
    radv = compute_RAVD(gt,pred)

    try:
        auc = compute_auc(gt,pred)
    except:
        auc = np.nan


    try:
        nsd = compute_nsd(gt,pred)
    except:
        nsd = np.nan

    # HD95/ASSD for 3D stacks (CHAOS):
    #   "voxel" mode -> one volumetric 3D computation per stack (default)
    #   "pixel" mode -> per 2D slice, averaged; slices where either mask is
    #                   empty return nan and are excluded, matching the
    #                   empty-image convention of the 2D datasets.
    # resolve spacing per use: 2D slices use the last two axes, 3D uses all (z padded with 1.0 if absent)
    sp2 = tuple(spacing[-2:]) if spacing is not None else None
    sp_full = None
    if spacing is not None:
        sp_full = tuple(spacing)
        if len(sp_full) < gt.ndim:
            sp_full = (1.0,) * (gt.ndim - len(sp_full)) + sp_full
        sp_full = sp_full[-gt.ndim:]

    try:
        if gt.ndim == 3 and distance_mode == "pixel":
            hd95_slices = [compute_hd95(gt[s], pred[s], sp2) for s in range(gt.shape[0])]
            if slice_store is not None:
                slice_store.setdefault("hd95", []).extend(hd95_slices)
            hd95 = np.nanmean(hd95_slices)
        else:
            hd95 = compute_hd95(gt,pred, sp_full)
    except:
        hd95 = np.nan

    try:
        if gt.ndim == 3 and distance_mode == "pixel":
            assd_slices = [compute_assd(gt[s], pred[s], sp2) for s in range(gt.shape[0])]
            if slice_store is not None:
                slice_store.setdefault("assd", []).extend(assd_slices)
            assd = np.nanmean(assd_slices)
        else:
            assd = compute_assd(gt,pred, sp_full)
    except:
        assd = np.nan

    met = [dsc,radv,auc,nsd,hd95,assd]
    # pdb.set_trace()

    return met


def effective_spacing(cfg, dataset, img_name):
    """
    Physical spacing (mm) of one test item ON THE EVALUATION GRID (cfg.img_size), or None
    when cfg.distance_units != "mm" (pixel/voxel units).
    Config keys: distance_units = mm | px ; spacing_y, spacing_x (native mm/px);
                 spacing_z (slice spacing, 3D only); native_h, native_w (optional, else read
                 from the original image file in dataset.image_dir).
    Effective spacing = native spacing x (native size / grid size), per axis, per image.
    """
    if getattr(cfg, "distance_units", "px") != "mm":
        return None
    H, W = int(cfg.img_size[0]), int(cfg.img_size[1])
    h0, w0 = getattr(cfg, "native_h", None), getattr(cfg, "native_w", None)
    if not (h0 and w0):
        img_dir = getattr(dataset, "image_dir", None)
        path = os.path.join(img_dir, img_name) if img_dir and img_name else None
        if path and os.path.isfile(path):
            from PIL import Image
            w0, h0 = Image.open(path).size          # lazy: header only
        else:
            h0, w0 = H, W                           # unknown native size -> assume grid = native
    sy = float(cfg.spacing_y) * h0 / H
    sx = float(cfg.spacing_x) * w0 / W
    sz = getattr(cfg, "spacing_z", None)
    return (float(sz), sy, sx) if sz else (sy, sx)

