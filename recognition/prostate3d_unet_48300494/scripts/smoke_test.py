"""
Smoke test for dataset and model: loads first sample from generated train split
and performs a forward pass through UNet3D. Prints shapes and a small set of
unique label values.
"""
import os
import sys

# Ensure project root is on path
sys.path.insert(0, os.getcwd())

from recognition.prostate3d_unet_48300494.dataset import Prostate3DDataset
from recognition.prostate3d_unet_48300494.modules import UNet3D
import numpy as np
import torch

train_list = os.path.join("recognition", "prostate3d_unet_48300494", "splits", "train_list.txt")
root_img = os.path.join("recognition", "prostate3d_unet_48300494", "data", "semantic_MRs_anon")
root_lbl = os.path.join("recognition", "prostate3d_unet_48300494", "data", "semantic_labels_anon")

print("Train list:", train_list)
if not os.path.exists(train_list):
    raise SystemExit("train_list.txt not found")

ds = Prostate3DDataset(train_list, root_img, root_lbl, augment=False)
print("Dataset length:", len(ds))
img, lbl = ds[0]
print("Image shape:", img.shape, "dtype:", img.dtype)
print("Label shape:", lbl.shape)
uniq = np.unique(lbl.numpy())
print("Unique label values (first 20):", uniq[:20])

# Model forward pass
model = UNet3D(in_channels=1, num_classes=5, base_channels=32)
model.eval()
with torch.no_grad():
    batch = img.unsqueeze(0)  # (1,1,D,H,W)
    logits = model(batch)
    print("Logits shape:", logits.shape)
print("Smoke test completed successfully.")

