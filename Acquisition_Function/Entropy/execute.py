import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Acquisition_Function/

import torch
from Entropy.floating_region import FloatingRegionScore
from Entropy.spatial_purity import SpatialPurity
from Entropy.loss.negative_learning_loss import NegativeLearningLoss,Pseudo_label_Loss

import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import pdb
import cv2
import os

import matplotlib.pyplot as plt


BCE_criterion = torch.nn.BCELoss()

negative_criterion = NegativeLearningLoss(threshold=0.05)


sup_criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
def loss_Entropy(img, pred, tgt_mask,cfg):
    sup_loss = sup_criterion(pred, tgt_mask)
    return sup_loss





def get_ROI_Entropy( model,image,tgt_mask,cfg):


    device = cfg.device

    model.eval()
    with torch.no_grad():
        primary_pred,feature = model(image,require_feature=True)
        # primary_pred = model(image)

    primary_pred = F.softmax(primary_pred,dim=1)
    pred = torch.argmax(primary_pred,dim=1)
    foreground_mask = (pred!=cfg.bg_label).float()

    floating_region_score = FloatingRegionScore(in_channels=cfg.num_classes, size=cfg.patch_size).to(device)
    per_region_pixels = cfg.patch_size ** 2
    active_radius = cfg.patch_size//2
    mask_radius = cfg.patch_size

    # pdb.set_trace()
    score_stack, purity_stack, entropy_stack = [], [], []

    for i1 in range(primary_pred.shape[0]):
        score_slice, purity, entropy = floating_region_score(torch.unsqueeze(primary_pred[i1],dim=0))

        boundary = extract_boundary_2d(foreground_mask[i1], radius=4)

        score_stack.append(score_slice.cpu().detach().numpy())
        purity_stack.append(purity.cpu().detach().numpy())
        entropy_stack.append(entropy.cpu().detach().numpy())

    # score_stack = np.array(score_stack)
    # purity_stack = np.array(purity_stack)
    # entropy_stack = np.array(entropy_stack)

    score_stack = np.array(entropy_stack)

    n_region = cfg.n_budget_box
    All_boxes = []

    for idx in range(score_stack.shape[0]):

        score_slice = score_stack[idx]
        slice_boxes = []

        temp = copy.deepcopy(score_slice)
        score_slice = np.clip(score_slice,0,1000)

        score_slice = (score_stack[idx]-np.min(score_stack[idx]))/(np.max(score_stack[idx])-np.min(score_stack[idx])+ 1e-8)


        tgt_mask_np = copy.deepcopy(tgt_mask[idx].cpu().numpy())
        score_slice[tgt_mask_np!=255] = -float('inf')

    
        for pixel in range(n_region):

            val_max = np.max(score_slice)
            loc_max = np.where(score_slice==val_max)
            loc_max = [loc_max[0][0],loc_max[1][0]]

            h, w = score_slice.shape

            active_start_x = np.clip(loc_max[0] - active_radius, 0, h)
            active_end_x = np.clip(loc_max[0] + active_radius + 1, 0, h)

            active_start_y = np.clip(loc_max[1] - active_radius, 0, w)
            active_end_y = np.clip(loc_max[1] + active_radius + 1, 0, w)

            mask_start_x = np.clip(loc_max[0] - mask_radius, 0, h)
            mask_end_x = np.clip(loc_max[0] + mask_radius + 1, 0, h)

            mask_start_y = np.clip(loc_max[1] - mask_radius, 0, w)
            mask_end_y = np.clip(loc_max[1] + mask_radius + 1, 0, w)

            # mask[active_start_x:active_end_x,active_start_y:active_end_y] = 1
            score_slice[mask_start_x:mask_end_x,mask_start_y:mask_end_y] = -float('inf')
            # slice_boxes.append([active_start_x, active_start_y, active_end_x, active_end_y])
            slice_boxes.append([int(active_start_y), int(active_start_x), int(active_end_y), int(active_end_x)])


        All_boxes.append(slice_boxes)

    model.train()

    return All_boxes





def draw_and_save_boxes(image_tensor, boxes, save_dir="output_boxes", prefix="img"):
    """
    Draws bounding boxes on images and saves them.

    Args:
        image_tensor (torch.Tensor): Tensor of shape (batch, channels, height, width).
        boxes (list): List of list of boxes per image. Each box: [x1, y1, x2, y2].
        save_dir (str): Directory to save images.
        prefix (str): Prefix for saved image filenames.
    """
    os.makedirs(save_dir, exist_ok=True)
    images = image_tensor.cpu().numpy()
    batch_size = images.shape[0]
    for idx in range(batch_size):
        img = images[idx]
        # Convert from CHW to HWC and scale to [0,255]
        img = np.transpose(img, (1, 2, 0))
        img = (img * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        for box in boxes[idx]:
            x1, y1, x2, y2 = box
            cv2.rectangle(img, (y1, x1), (y2-1, x2-1), (0, 255, 0), 2)
        cv2.imwrite(os.path.join(save_dir, f"{prefix}_{idx}.png"), img)


def extract_boundary_2d(mask, radius=1):
    """
    Extract boundary from a 2D binary mask using a morphological ball (disk).

    Args:
        mask (torch.Tensor or np.ndarray): binary mask of shape (H, W)
        radius (int): radius of the structuring element (ball size)

    Returns:
        torch.Tensor: binary boundary map (1 on boundary, 0 elsewhere)
    """
    if torch.is_tensor(mask):
        mask = mask.detach().cpu().numpy()
    mask = (mask > 0.5).astype(np.uint8)

    # Create a circular (disk) structuring element
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius + 1, 2*radius + 1))

    # Erode to get inner region
    eroded = cv2.erode(mask, kernel)

    # Subtract to get boundary
    boundary = mask - eroded
    boundary = np.clip(boundary, 0, 1)

    return torch.from_numpy(boundary).float()

if __name__ == "__main__":

    class DummyModel(torch.nn.Module):
        def __init__(self, num_classes):
            super().__init__()
            self.num_classes = num_classes

        def forward(self, x, require_feature=False):
            batch, c, h, w = x.shape
            primary_pred = torch.rand(batch, self.num_classes, h, w).to(x.device)
            # Simulate feature as a list of tensors per layer (as expected by get_weight_map)
            feature = [torch.rand(batch, 64, h, w).to(x.device)]
            concat_feature = torch.rand(batch, 128, h, w).to(x.device)
            if require_feature:
                return primary_pred, feature, concat_feature
            else:
                return primary_pred

    class DummyCfg:
        def __init__(self):
            self.device = None
            self.num_classes = 10
            self.patch_size = 21
            self.budget = 10
            self.n_budget_box = None


    cfg = DummyCfg()
    cfg.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image = torch.rand(5, 3, 448, 448).to(cfg.device)  # Example image tensor
    model = DummyModel(num_classes=cfg.num_classes).to(cfg.device)
    boxes = get_ROI_Entropy(model, image, cfg)
    draw_and_save_boxes(image, boxes, save_dir="output_boxes", prefix="img")
    print("Selected ROI boxes:", boxes)


    # Example usage of loss_Entropy
    primary_pred = model(image)
    # Simulate target mask (random integers in [0, num_classes-1], shape: (batch, height, width))
    tgt_mask = torch.randint(0, cfg.num_classes, (image.shape[0], image.shape[2], image.shape[3])).to(cfg.device)
    loss = loss_Entropy(image, primary_pred, tgt_mask)
    print("Calculated loss:", loss.item())