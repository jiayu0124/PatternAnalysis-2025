"""
generate_patient_splits.py
--------------------------
Generate train/val/test splits at the patient level.

This script groups all case files by patient id (e.g. "Case_010") and
ensures that all weeks for the same patient go into the same split. It
writes per-sample identifiers (e.g. "Case_010_Week3") into the
`splits/` directory.

Usage (from project root):

    python recognition\prostate3d_unet_48300494\scripts\generate_patient_splits.py \
        --root_img recognition\prostate3d_unet_48300494\data\semantic_MRs_anon \
        --root_lbl recognition\prostate3d_unet_48300494\data\semantic_labels_anon

"""
import argparse
import os
import re
import random
from collections import defaultdict
from typing import Dict, List, Set

# Patient regex matches the Case number (e.g. Case_010)
PATIENT_RE = re.compile(r"^(Case_\d+)")
SAMPLE_RE = re.compile(r"^(Case_\d+_Week\d+)")


def collect_samples(root_img: str, root_lbl: str) -> Dict[str, List[str]]:
    """Return a mapping patient_id -> list of sample_ids (Case_xxx_Weeky).

    Only samples present in both image and label directories are included.
    """
    img_files = [f for f in os.listdir(root_img) if f.lower().endswith('.nii') or f.lower().endswith('.nii.gz')]
    lbl_files = [f for f in os.listdir(root_lbl) if f.lower().endswith('.nii') or f.lower().endswith('.nii.gz')]
    # Extract sample ids from image filenames
    img_sample_ids: Set[str] = set()
    for fn in img_files:
        m = SAMPLE_RE.search(fn)
        if m:
            img_sample_ids.add(m.group(1))
    # For labels we try to extract sample ids by looking for the SAMPLE_RE inside filename
    lbl_sample_ids: Set[str] = set()
    for fn in lbl_files:
        m = SAMPLE_RE.search(fn)
        if m:
            lbl_sample_ids.add(m.group(1))
    # Now consider all sample ids that exist in both
    common_samples = sorted(list(img_sample_ids & lbl_sample_ids))
    # Group by patient
    patient_map: Dict[str, List[str]] = defaultdict(list)
    for sample in common_samples:
        pm = PATIENT_RE.match(sample)
        if pm:
            patient = pm.group(1)
            patient_map[patient].append(sample)
    return patient_map


def write_split_files(out_dir: str, train_samples: List[str], val_samples: List[str], test_samples: List[str]) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'train_list.txt'), 'w') as f:
        for s in train_samples:
            f.write(s + '\n')
    with open(os.path.join(out_dir, 'val_list.txt'), 'w') as f:
        for s in val_samples:
            f.write(s + '\n')
    with open(os.path.join(out_dir, 'test_list.txt'), 'w') as f:
        for s in test_samples:
            f.write(s + '\n')


def generate_patient_splits(root_img: str, root_lbl: str, out_dir: str, train_ratio: float, val_ratio: float, seed: int) -> None:
    patient_map = collect_samples(root_img, root_lbl)
    if not patient_map:
        raise RuntimeError('No matched samples found in the provided directories')
    patients = sorted(patient_map.keys())
    random.seed(seed)
    random.shuffle(patients)
    n = len(patients)
    n_train = max(1, int(n * train_ratio))
    n_val = max(1, int(n * val_ratio))
    if n_train + n_val >= n:
        n_train = max(1, n - 2)
        n_val = max(1, (n - n_train) // 2)
    n_test = n - n_train - n_val
    if n_test < 0:
        n_test = 0
    train_patients = patients[:n_train]
    val_patients = patients[n_train:n_train + n_val]
    test_patients = patients[n_train + n_val:]
    # Expand to samples
    train_samples = [s for p in train_patients for s in sorted(patient_map[p])]
    val_samples = [s for p in val_patients for s in sorted(patient_map[p])]
    test_samples = [s for p in test_patients for s in sorted(patient_map[p])]
    write_split_files(out_dir, train_samples, val_samples, test_samples)
    print(f"Patients: {n} -> Train patients: {len(train_patients)}, Val patients: {len(val_patients)}, Test patients: {len(test_patients)}")
    print(f"Samples: Train {len(train_samples)}, Val {len(val_samples)}, Test {len(test_samples)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_img', required=True)
    parser.add_argument('--root_lbl', required=True)
    parser.add_argument('--out_dir', default=os.path.join(os.path.dirname(__file__), '..', 'splits'))
    parser.add_argument('--train_ratio', type=float, default=0.7)
    parser.add_argument('--val_ratio', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    generate_patient_splits(args.root_img, args.root_lbl, args.out_dir, args.train_ratio, args.val_ratio, args.seed)
