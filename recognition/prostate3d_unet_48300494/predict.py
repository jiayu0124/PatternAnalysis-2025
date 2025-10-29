"""
predict.py
-----------

This script loads a trained 3‑D U‑Net and evaluates it on a list of
cases.  It reports per‑class and mean Dice coefficients and can
optionally save the predicted segmentation volumes as NIfTI files.

Usage example::

    python predict.py \
        --root_img /home/groups/comp3710/HipMRI_Study_open/semantic_MRs \
        --root_lbl /home/groups/comp3710/HipMRI_Study_open/semantic_labels_only \
        --test_list splits/test_list.txt \
        --model outputs/best_model.pt \
        --save_dir outputs/predictions

Refer to the README for more information.
"""

import argparse
import os
from typing import List

import nibabel as nib  # type: ignore
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import Prostate3DDataset
from modules import UNet3D
from utils import dice_coefficients


def pad_collate(batch):
    # Reuse the same collate function as in training
    images, labels = zip(*batch)
    depths = [img.shape[1] for img in images]
    heights = [img.shape[2] for img in images]
    widths = [img.shape[3] for img in images]
    max_d, max_h, max_w = max(depths), max(heights), max(widths)
    padded_images = []
    padded_labels = []
    for img, lbl in zip(images, labels):
        d_pad = max_d - img.shape[1]
        h_pad = max_h - img.shape[2]
        w_pad = max_w - img.shape[3]
        padded_img = torch.nn.functional.pad(img, (0, w_pad, 0, h_pad, 0, d_pad))
        padded_lbl = torch.nn.functional.pad(lbl, (0, w_pad, 0, h_pad, 0, d_pad))
        padded_images.append(padded_img)
        padded_labels.append(padded_lbl)
    return torch.stack(padded_images), torch.stack(padded_labels)


def evaluate(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    test_ds = Prostate3DDataset(args.test_list, args.root_img, args.root_lbl, augment=False)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=pad_collate)
    model = UNet3D(in_channels=1, num_classes=args.num_classes, base_channels=args.base_channels)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.to(device)
    model.eval()
    all_dice: List[List[float]] = []
    # Optional saving directory
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            dice = dice_coefficients(outputs, labels, args.num_classes)
            all_dice.append(dice)
            if args.save_dir:
                # Save predicted segmentation as NIfTI
                preds = torch.argmax(outputs, dim=1)[0].cpu().numpy().astype(np.uint8)
                # The original affine can be retrieved by reloading the corresponding image
                case_id = test_ds.case_ids[i]
                ref_nii = nib.load(os.path.join(args.root_img, f"{case_id}.nii.gz"))
                pred_nii = nib.Nifti1Image(preds, ref_nii.affine)
                nib.save(pred_nii, os.path.join(args.save_dir, f"{case_id}_pred.nii.gz"))
                print(f"Saved prediction for {case_id} to {args.save_dir}")
    # Compute mean dice over all cases
    if all_dice:
        all_dice_arr = np.array(all_dice)  # shape (N_cases, num_classes)
        mean_dice_per_class = all_dice_arr.mean(axis=0)
        mean_dice = mean_dice_per_class.mean()
        print("Per‑class Dice coefficients:")
        for c, d in enumerate(mean_dice_per_class):
            print(f"  Class {c}: {d:.4f}")
        print(f"Mean Dice over all classes: {mean_dice:.4f}")
    else:
        print("No test cases found.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained 3‑D UNet on the test set")
    parser.add_argument("--root_img", type=str, required=True, help="Directory containing MRI volumes")
    parser.add_argument("--root_lbl", type=str, required=True, help="Directory containing segmentation labels")
    parser.add_argument("--test_list", type=str, required=True, help="Text file listing test cases")
    parser.add_argument("--model", type=str, required=True, help="Path to the trained model (.pt) file")
    parser.add_argument("--save_dir", type=str, default=None, help="Directory to save predicted volumes (optional)")
    parser.add_argument("--num_classes", type=int, default=5, help="Number of segmentation classes")
    parser.add_argument("--base_channels", type=int, default=32, help="Number of base channels in UNet")
    parser.add_argument("--no_cuda", action="store_true", help="Force evaluation on CPU even if CUDA is available")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)