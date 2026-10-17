"""Datasets.

Layout expected on disk: <data_dir>/<split>/data/<name> and <data_dir>/<split>/labels/<name>.

AL_mode=True adds a per-image "labelled pool" mask (AL_dir/<name>.tif).
It starts as all 255 (= unlabelled, ignored by the loss) and LINGUAL writes
corrected labels into it inside the selected boxes every round.
"""
import os
import shutil

import numpy as np
import tifffile
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IGNORE = 255


def _label_scale(dataset_name):
    # CHAOS labels are stored as multiples of 63, binary datasets as 0/255
    return 63 if dataset_name == "CHAOS" else 255


class _Base(Dataset):
    def __init__(self, dataset_name, data_dir, split="train", augmentation=False, img_size=(448, 448)):
        self.dataset_name = dataset_name
        self.image_dir = os.path.join(data_dir, split, "data")
        self.mask_dir = os.path.join(data_dir, split, "labels")
        self.image_files = sorted(os.listdir(self.image_dir))
        self.image_size = img_size

        self.resize = transforms.Resize(img_size)
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
        self.aug_transform = transforms.RandomChoice([
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.RandomVerticalFlip(p=1.0),
            transforms.RandomRotation(degrees=15),
        ]) if augmentation else None

    def __len__(self):
        return len(self.image_files)

    def _prepare(self, image, mask):
        if self.aug_transform:
            seed = np.random.randint(2147483647)
            torch.manual_seed(seed)
            image = self.aug_transform(image)
            torch.manual_seed(seed)
            mask = self.aug_transform(mask)
        image = self.normalize(self.to_tensor(self.resize(image)))
        mask = mask.resize(self.image_size, Image.NEAREST)
        mask = torch.from_numpy(np.array(mask, dtype=np.uint8)).float() / _label_scale(self.dataset_name)
        return image, mask


class Seg_Dataset(_Base):
    """2D images. Returns (image, gt, name) or, in AL_mode, (image, gt, name, pool_mask)."""

    def __init__(self, dataset_name, data_dir, split="train", augmentation=False, AL_mode=False,
                 Exp_name="Exp", img_size=(448, 448)):
        super().__init__(dataset_name, data_dir, split, augmentation, img_size)
        self.AL_mode = AL_mode
        if AL_mode:
            self.AL_dir = os.path.join(data_dir, split, "AL_mask", Exp_name)
            if os.path.exists(self.AL_dir):
                shutil.rmtree(self.AL_dir)
            os.makedirs(self.AL_dir, exist_ok=True)
            empty = np.full(img_size, IGNORE, dtype=np.uint8)
            for name in self.image_files:
                tifffile.imwrite(os.path.join(self.AL_dir, name), empty)

    def __getitem__(self, idx):
        name = self.image_files[idx]
        image = Image.open(os.path.join(self.image_dir, name)).convert("RGB")
        mask = Image.open(os.path.join(self.mask_dir, name)).convert("L")
        image, mask = self._prepare(image, mask)
        if self.AL_mode:
            return image, mask, name, tifffile.imread(os.path.join(self.AL_dir, name))
        return image, mask, name


class Stack_Dataset(_Base):
    """3D tif stacks (CHAOS test). Returns (slices, gt_slices, name)."""

    def __getitem__(self, idx):
        name = self.image_files[idx]
        image_stack = tifffile.imread(os.path.join(self.image_dir, name))
        mask_stack = tifffile.imread(os.path.join(self.mask_dir, name))
        images, masks = [], []
        for i in range(image_stack.shape[0]):
            image, mask = self._prepare(Image.fromarray(image_stack[i]).convert("RGB"),
                                        Image.fromarray(mask_stack[i]).convert("L"))
            images.append(image)
            masks.append(mask)
        return torch.stack(images), torch.stack(masks), name


def load_rgb(image_path, img_size):
    """Image as the executors see it: RGB PIL at img_size (tif is min-max normalised first)."""
    if image_path.endswith((".tif", ".tiff")):
        image = tifffile.imread(image_path).astype(np.float64)
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        image = Image.fromarray((image * 255).astype(np.uint8)).convert("RGB")
    else:
        image = Image.open(image_path).convert("RGB")
    return image.resize(img_size, Image.BILINEAR)
