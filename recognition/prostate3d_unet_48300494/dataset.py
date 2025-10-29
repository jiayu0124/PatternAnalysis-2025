"""
dataset.py
-------------

This module defines a PyTorch `Dataset` for the COMP3710 project on
3‑D prostate segmentation.  It reads paired MRI volumes and
corresponding label volumes stored in NIfTI format.  The dataset
normalises image intensities on a per‑volume basis and optionally
applies simple random data augmentation (random flips along each
spatial axis).  Labels are returned as integer class indices without
one‑hot encoding; the downstream loss function should handle
class indices directly (e.g. `torch.nn.CrossEntropyLoss`).

The expected directory layout is as follows::

    ├── semantic_MRs
    │   ├── case0001.nii.gz
    │   ├── case0002.nii.gz
    │   └── ...
    └── semantic_labels_only
        ├── case0001.nii.gz
        ├── case0002.nii.gz
        └── ...

Additionally you should provide a plain text file listing the case
identifiers (without extension) for the training, validation and test
splits.  For example `train_list.txt` might contain::

    case0001
    case0002
    case0003
    ...

See the accompanying README for details on preparing these lists.
"""

import os
import random
from typing import List, Tuple

import nibabel as nib  # type: ignore
import numpy as np
import torch
from torch.utils.data import Dataset


def normalise(volume: np.ndarray) -> np.ndarray:
    """Normalise a 3‑D volume by subtracting its mean and dividing by its
    standard deviation.  A small constant is added to the denominator to
    avoid division by zero.

    Parameters
    ----------
    volume : np.ndarray
        The input image volume (float32).

    Returns
    -------
    np.ndarray
        The normalised volume.
    """
    mean = volume.mean()
    std = volume.std()
    return (volume - mean) / (std + 1e-8)


def random_flip(img: np.ndarray, lbl: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Randomly flip the image and label volumes along each axis with 50% probability.

    Parameters
    ----------
    img : np.ndarray
        Input image volume.
    lbl : np.ndarray
        Corresponding label volume.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        The possibly flipped image and label volumes.
    """
    for axis in range(3):
        if random.random() < 0.5:
            img = np.flip(img, axis=axis).copy()
            lbl = np.flip(lbl, axis=axis).copy()
    return img, lbl


class Prostate3DDataset(Dataset):
    """PyTorch dataset for 3‑D prostate MRI and segmentation labels.

    Each item returned by this dataset is a tuple `(image, label)` where
    `image` is a float32 tensor of shape `(1, D, H, W)` and `label`
    is a long tensor of shape `(D, H, W)`.  The voxel spacing and
    affine transforms are ignored for simplicity; if you need to
    preserve spacing you should adapt this class accordingly.

    Parameters
    ----------
    list_file : str
        Path to a text file containing one case identifier per line (without
        extension).  The dataset uses these identifiers to locate the
        corresponding image and label files under `root_img` and
        `root_lbl`.
    root_img : str
        Directory containing the MRI volumes in NIfTI format.
    root_lbl : str
        Directory containing the segmentation labels in NIfTI format.
    augment : bool, optional
        If True, apply simple random flip augmentation.  Defaults to False.
    """

    def __init__(self, list_file: str, root_img: str, root_lbl: str, augment: bool = False) -> None:
        with open(list_file, "r") as f:
            self.case_ids: List[str] = [line.strip() for line in f if line.strip()]
        self.root_img = root_img
        self.root_lbl = root_lbl
        self.augment = augment

    def __len__(self) -> int:
        return len(self.case_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        case_id = self.case_ids[idx]
        img_path = os.path.join(self.root_img, f"{case_id}.nii.gz")
        lbl_path = os.path.join(self.root_lbl, f"{case_id}.nii.gz")
        # Load image and label volumes
        img = nib.load(img_path).get_fdata(caching="unchanged").astype(np.float32)
        lbl = nib.load(lbl_path).get_fdata(caching="unchanged").astype(np.int64)
        # Remove singleton dimensions if present (sometimes 4D with last dim size 1)
        if img.ndim == 4 and img.shape[-1] == 1:
            img = img[..., 0]
        if lbl.ndim == 4 and lbl.shape[-1] == 1:
            lbl = lbl[..., 0]
        # Normalise image intensities
        img = normalise(img)
        # Apply augmentation
        if self.augment:
            img, lbl = random_flip(img, lbl)
        # Convert to torch tensors
        image_tensor = torch.from_numpy(img).unsqueeze(0).float()  # shape (1, D, H, W)
        label_tensor = torch.from_numpy(lbl).long()  # shape (D, H, W)
        return image_tensor, label_tensor