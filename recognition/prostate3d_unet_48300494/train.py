"""
train.py
---------

This script trains a 3‑D U‑Net on the prostate segmentation problem using
PyTorch.  It performs the following steps:

1. Loads a training and validation split defined in text files.
2. Instantiates the `Prostate3DDataset` defined in `dataset.py`.
3. Creates a DataLoader with a custom collate function that pads
   volumes within each batch to a common spatial size.
4. Defines a 3‑D U‑Net model, a mixed Dice + Cross‑Entropy loss and
   an Adam optimiser.
5. Trains for a configurable number of epochs, computing mean Dice on
   the validation set after each epoch.  The best model (highest mean
   Dice) is saved to disk.
6. Saves a plot of the training/validation loss and mean Dice curves.

Run this script from the project root directory.  See the README
for example commands.
"""

import argparse
import os
from contextlib import nullcontext
from typing import Tuple

import matplotlib
matplotlib.use("Agg")  # Use a non‑interactive backend
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from dataset import Prostate3DDataset  # noqa: E402
from modules import UNet3D  # noqa: E402
from utils import DiceLoss, dice_coefficients  # noqa: E402
import csv


def pad_collate(batch: Tuple[Tuple[torch.Tensor, torch.Tensor], ...]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Collate function that pads all images and labels in a batch to the same size.

    The largest depth/height/width across the batch is used, and padding
    is applied to the end of each dimension.  This allows volumes of
    different shapes to be stacked into a single tensor.  Labels are
    padded with zeros (background class).

    Parameters
    ----------
    batch : list of tuples
        Each element is a `(image, label)` pair returned by the dataset.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        A tuple `(images, labels)` where `images` has shape
        `(B, 1, D_max, H_max, W_max)` and `labels` has shape
        `(B, D_max, H_max, W_max)`.
    """
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
        padded_img = F.pad(img, (0, w_pad, 0, h_pad, 0, d_pad))
        padded_lbl = F.pad(lbl, (0, w_pad, 0, h_pad, 0, d_pad))
        padded_images.append(padded_img)
        padded_labels.append(padded_lbl)
    return torch.stack(padded_images), torch.stack(padded_labels)


def _write_metrics_csv(path: str, epoch: int, train_loss: float, val_loss: float, val_dice: float) -> None:
    header = ["epoch", "train_loss", "val_loss", "val_dice"]
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow([epoch, f"{train_loss:.6f}", f"{val_loss:.6f}", f"{val_dice:.6f}"])


def _write_metrics_per_class_csv(path: str, epoch: int, per_class: np.ndarray, mean_dice: float, min_dice: float, max_dice: float) -> None:
    """Write per-class Dice metrics to a separate CSV to avoid header mismatch with legacy metrics.csv.
    Columns: epoch, mean, min, max, dice_c0..dice_c{K}.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    num_classes = per_class.shape[0]
    header = ["epoch", "mean", "min", "max"] + [f"dice_c{i}" for i in range(num_classes)]
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        row = [epoch, f"{mean_dice:.6f}", f"{min_dice:.6f}", f"{max_dice:.6f}"] + [f"{float(v):.6f}" for v in per_class.tolist()]
        writer.writerow(row)


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    # Enable cuDNN autotuner for potentially faster convs on fixed shapes
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # Create datasets
    train_ds = Prostate3DDataset(args.train_list, args.root_img, args.root_lbl, augment=True)
    val_ds = Prostate3DDataset(args.val_list, args.root_img, args.root_lbl, augment=False)

    # DataLoaders (use pin_memory when on CUDA for faster host->device copies)
    pin_mem = device.type == "cuda"
    # Visible runtime info
    print(f"Using device: {device}. pin_memory={pin_mem}")
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=pad_collate,
        pin_memory=pin_mem,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=pad_collate,
        pin_memory=pin_mem,
    )

    # Fast preflight: optionally scan a small number of samples to validate label range
    if hasattr(args, 'skip_label_scan') and not args.skip_label_scan:
        scan_n = min(getattr(args, 'label_scan_limit', 2), len(train_ds)) if getattr(args, 'label_scan_limit', 2) > 0 else 0
        max_label_seen = -1
        print(f"[Preflight] Scanning up to {scan_n} training samples for label range validation...")
        for i in range(scan_n):
            _, lbl = train_ds[i]
            _max = int(lbl.max().item())
            if _max > max_label_seen:
                max_label_seen = _max
            if i == 0 or i == scan_n - 1:
                print(f"[Preflight] sample {i} max label: {_max}")
        if max_label_seen >= 0 and max_label_seen >= args.num_classes:
            print(
                f"[Config Error] Detected label index {max_label_seen} but num_classes={args.num_classes}. "
                f"Please re-run with --num_classes {max_label_seen + 1}. Aborting.")
            return
    elif hasattr(args, 'skip_label_scan') and args.skip_label_scan:
        print("[Preflight] Skipped label range validation (per --skip_label_scan)")
    else:
        # Backward-compat: keep a minimal one-sample check if flags are absent
        try:
            _, _lbl = train_ds[0]
            _mx = int(_lbl.max().item())
            if _mx >= args.num_classes:
                print(f"[Config Error] Detected label index {_mx} but num_classes={args.num_classes}. "
                      f"Please re-run with --num_classes {_mx + 1}. Aborting.")
                return
        except Exception:
            pass

    # Model, loss, optimiser
    model = UNet3D(in_channels=1, num_classes=args.num_classes, base_channels=args.base_channels).to(device)
    criterion = DiceLoss(weight=args.dice_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # AMP scaler (only meaningful on CUDA)
    use_amp = device.type == "cuda"
    print(f"AMP enabled: {use_amp}")
    scaler = torch.amp.GradScaler(device="cuda") if use_amp else torch.amp.GradScaler(enabled=False)

    # Training history
    history = {"train_loss": [], "val_loss": [], "val_dice": []}
    best_dice = 0.0
    os.makedirs(args.out_dir, exist_ok=True)
    metrics_csv = os.path.join(args.out_dir, "metrics.csv")
    metrics_pc_csv = os.path.join(args.out_dir, "metrics_per_class.csv")

    # Training loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            # Non-blocking transfers for faster H2D when pin_memory=True
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            # Forward + loss with autocast on CUDA
            ctx = torch.amp.autocast(device_type="cuda") if use_amp else nullcontext()
            with ctx:
                outputs = model(images)
                loss = criterion(outputs, labels)
            # Backward + step via GradScaler when AMP is enabled
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * images.size(0)
        train_loss = running_loss / len(train_ds) if len(train_ds) > 0 else 0.0
        history["train_loss"].append(train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        # accumulate per-batch per-class dice
        per_class_batches: list[np.ndarray] = []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                ctx = torch.amp.autocast(device_type="cuda") if use_amp else nullcontext()
                with ctx:
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                pc_list = dice_coefficients(outputs, labels, args.num_classes)
                per_class_batches.append(np.array(pc_list, dtype=np.float32))
        val_loss = val_loss / len(val_ds) if len(val_ds) > 0 else 0.0

        # Aggregate per-class across batches
        if per_class_batches:
            per_class_mean = np.stack(per_class_batches, axis=0).mean(axis=0)
            min_dice = float(per_class_mean.min())
            max_dice = float(per_class_mean.max())
            mean_dice = float(per_class_mean.mean())
        else:
            per_class_mean = np.zeros((args.num_classes,), dtype=np.float32)
            min_dice = 0.0
            max_dice = 0.0
            mean_dice = 0.0

        history["val_loss"].append(val_loss)
        history["val_dice"].append(mean_dice)

        # Write metrics
        _write_metrics_csv(metrics_csv, epoch, train_loss, val_loss, mean_dice)
        _write_metrics_per_class_csv(metrics_pc_csv, epoch, per_class_mean, mean_dice, min_dice, max_dice)

        # Save best model
        if mean_dice > best_dice:
            best_dice = mean_dice
            torch.save(model.state_dict(), os.path.join(args.out_dir, "best_model.pt"))

        # Threshold status for readability
        thr = getattr(args, 'dice_threshold', 0.7)
        all_ok = bool((per_class_mean >= thr).all())
        status = "PASS" if all_ok else "FAIL"
        print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Dice(mean): {mean_dice:.4f} | min_c: {min_dice:.4f} | thr {thr} -> {status}")
        # Optional: print compact per-class vector (first few and last few if many)
        if args.num_classes <= 10:
            pcs = ", ".join(f"c{i}:{per_class_mean[i]:.3f}" for i in range(args.num_classes))
            print(f"Per-class Val Dice: [{pcs}]")
        else:
            pcs_head = ", ".join(f"c{i}:{per_class_mean[i]:.3f}" for i in range(5))
            pcs_tail = ", ".join(f"c{i}:{per_class_mean[i]:.3f}" for i in range(args.num_classes-5, args.num_classes))
            print(f"Per-class Val Dice: [{pcs_head}, ..., {pcs_tail}]")

    # Plot learning curves using the recorded history length (avoid relying on args.epochs)
    n_epochs_recorded = len(history["train_loss"])
    if n_epochs_recorded == 0:
        print("No training history recorded, skipping plot generation.")
        return
    epochs = range(1, n_epochs_recorded + 1)
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    # Use markers so single-epoch plots are visible
    ax1.plot(epochs, history["train_loss"], label="Train Loss", marker='o')
    ax1.plot(epochs, history["val_loss"], label="Val Loss", marker='o')
    ax1.legend(loc="upper left")
    # Create a second y‑axis for Dice
    ax2 = ax1.twinx()
    ax2.set_ylabel("Mean Dice")
    ax2.plot(epochs, history["val_dice"], label="Val Dice", linestyle="--", marker='o')
    ax2.legend(loc="upper right")
    plt.title("Training Curves")
    plt.tight_layout()
    curve_path = os.path.join(args.out_dir, "curves.png")
    plt.savefig(curve_path)
    plt.close(fig)
    print(f"Training complete. Best mean Dice: {best_dice:.4f}. Curves saved to {curve_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a 3-D UNet on the prostate dataset")
    parser.add_argument("--root_img", type=str, required=True, help="Directory containing MRI volumes")
    parser.add_argument("--root_lbl", type=str, required=True, help="Directory containing segmentation labels")
    parser.add_argument("--val_list", type=str, required=True, help="Text file listing validation cases")
    parser.add_argument("--train_list", type=str, required=True, help="Text file listing training cases")
    parser.add_argument("--out_dir", type=str, default="outputs", help="Directory to save models and plots")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (number of volumes per batch)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--num_classes", type=int, default=6, help="Number of segmentation classes (background + organs)")
    parser.add_argument("--base_channels", type=int, default=32, help="Number of base channels in UNet")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    parser.add_argument("--dice_weight", type=float, default=1.0, help="Weight of Dice loss (0–1).  1 uses only Dice loss, 0 only CE.")
    parser.add_argument("--no_cuda", action="store_true", help="Force training on CPU even if CUDA is available")
    # New fast-preflight controls
    parser.add_argument("--skip_label_scan", action="store_true", help="Skip label range preflight to start training immediately")
    parser.add_argument("--label_scan_limit", type=int, default=2, help="Number of training samples to scan for label validation (0 disables)")
    # New dice threshold for status line
    parser.add_argument("--dice_threshold", type=float, default=0.7, help="Threshold for per-class Dice PASS/FAIL indication")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)