"""
generate_splits.py
------------------
Utility to generate train/val/test split files for the prostate 3D
UNet project. It scans the image and label directories, extracts a
common case identifier (e.g. `Case_004_Week0`) and writes three files
into the `splits/` directory: `train_list.txt`, `val_list.txt` and
`test_list.txt`.

By default it uses a 70/15/15 random split (with a fixed seed for
reproducibility). If there are too few samples the script will ensure
that each split has at least one element when possible.

Usage (from project root):

    python recognition\prostate3d_unet_48300494\scripts\generate_splits.py \
        --root_img recognition\prostate3d_unet_48300494\data\semantic_MRs_anon \
        --root_lbl recognition\prostate3d_unet_48300494\data\semantic_labels_anon

"""

import argparse
import os
import re
import random
from typing import List, Set


ID_RE = re.compile(r"^(Case_\d+_Week\d+)")


def extract_ids_from_filenames(filenames: List[str]) -> Set[str]:
    ids = set()
    for fn in filenames:
        m = ID_RE.match(fn)
        if m:
            ids.add(m.group(1))
    return ids


def write_list(file_path: str, ids: List[str]) -> None:
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as f:
        for cid in ids:
            f.write(cid + "\n")


def generate_splits(root_img: str, root_lbl: str, out_dir: str, train_ratio: float, val_ratio: float, seed: int) -> None:
    img_files = [f for f in os.listdir(root_img) if f.lower().endswith('.nii') or f.lower().endswith('.nii.gz')]
    lbl_files = [f for f in os.listdir(root_lbl) if f.lower().endswith('.nii') or f.lower().endswith('.nii.gz')]
    img_ids = extract_ids_from_filenames(img_files)
    lbl_ids = extract_ids_from_filenames(lbl_files)
    common_ids = sorted(list(img_ids & lbl_ids))

    if not common_ids:
        raise RuntimeError(f"No matching case IDs found between {root_img} and {root_lbl}")

    random.seed(seed)
    random.shuffle(common_ids)

    n = len(common_ids)
    n_train = max(1, int(n * train_ratio))
    n_val = max(1, int(n * val_ratio))
    # Ensure we don't exceed available samples
    if n_train + n_val >= n:
        n_train = max(1, n - 2)
        n_val = max(1, (n - n_train) // 2)
    n_test = n - n_train - n_val
    if n_test < 1:
        # adjust to ensure at least one in test if possible
        if n >= 3:
            n_test = 1
            if n_train > n_val:
                n_train -= 1
            else:
                n_val -= 1
        else:
            n_test = 0

    train_ids = common_ids[:n_train]
    val_ids = common_ids[n_train:n_train + n_val]
    test_ids = common_ids[n_train + n_val:]

    splits_dir = os.path.join(out_dir)
    os.makedirs(splits_dir, exist_ok=True)
    write_list(os.path.join(splits_dir, "train_list.txt"), train_ids)
    write_list(os.path.join(splits_dir, "val_list.txt"), val_ids)
    write_list(os.path.join(splits_dir, "test_list.txt"), test_ids)

    print(f"Found {n} matched cases. Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_img", required=True, help="Directory containing image .nii(.gz) files")
    parser.add_argument("--root_lbl", required=True, help="Directory containing label .nii(.gz) files")
    parser.add_argument("--out_dir", default=os.path.join(os.path.dirname(__file__), '..', 'splits'), help="Directory to write splits into (default ./splits)")
    parser.add_argument("--train_ratio", type=float, default=0.7, help="Fraction of data to use for training")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Fraction of data to use for validation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()
    generate_splits(args.root_img, args.root_lbl, args.out_dir, args.train_ratio, args.val_ratio, args.seed)

