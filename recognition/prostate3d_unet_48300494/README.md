# Prostate 3D UNet (ID: 48300494)

This project implements a 3D U-Net to segment HipMRI prostate radiotherapy MR volumes into 6 classes (background + 5 structures). It fulfils Project 7 requirements: on the held-out test split, all classes achieve Dice ≥ 0.7.

## Results (Test set)
Per-class Dice (averaged over test set):
- Class 0 (background): 0.9922
- Class 1: 0.9783
- Class 2: 0.9050
- Class 3: 0.9268
- Class 4: 0.8490
- Class 5: 0.8340

Mean Dice over all classes: 0.9142

Curves: see `outputs/curves.png` (from the 50-epoch full training). Best checkpoint saved at `outputs/best_model.pt`.

## Environment
- Python 3.12
- PyTorch (CUDA optional, AMP enabled if CUDA is available)
- nibabel, numpy, matplotlib

Install (Conda or pip):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install nibabel numpy matplotlib
```

## Data layout
```
recognition/prostate3d_unet_48300494/
  data/
    semantic_MRs_anon/            # MR volumes (NIfTI)
    semantic_labels_anon/         # Label volumes (NIfTI)
  splits/
    train_list.txt
    val_list.txt
    test_list.txt
```

## Train
Full training (50 epochs, base_channels=16):
```bash
python -u recognition/prostate3d_unet_48300494/train.py \
  --root_img recognition/prostate3d_unet_48300494/data/semantic_MRs_anon \
  --root_lbl recognition/prostate3d_unet_48300494/data/semantic_labels_anon \
  --train_list recognition/prostate3d_unet_48300494/splits/train_list.txt \
  --val_list recognition/prostate3d_unet_48300494/splits/val_list.txt \
  --out_dir recognition/prostate3d_unet_48300494/outputs \
  --epochs 50 \
  --base_channels 16 \
  --num_workers 0
```

## Evaluate (Test set)
```bash
python -u recognition/prostate3d_unet_48300494/predict.py \
  --root_img recognition/prostate3d_unet_48300494/data/semantic_MRs_anon \
  --root_lbl recognition/prostate3d_unet_48300494/data/semantic_labels_anon \
  --test_list recognition/prostate3d_unet_48300494/splits/test_list.txt \
  --model recognition/prostate3d_unet_48300494/outputs/best_model.pt \
  --num_classes 6 \
  --base_channels 16
```

Optionally save predictions:
```bash
python -u recognition/prostate3d_unet_48300494/predict.py ... --save_dir recognition/prostate3d_unet_48300494/outputs/preds
```

## Notes
- Mixed precision (AMP) and cuDNN benchmark are enabled automatically on CUDA.
- Data loading pads within batch to the max shape via a custom collate function.
- Per-class Dice metrics are written to `outputs/metrics_per_class.csv`; curves to `outputs/curves.png`.

## Acknowledgements
- Based on standard UNet3D for volumetric segmentation; data provided by HipMRI Study.

## AI Assistance Declaration
- In line with the course policy (Sec. 1.5), AI tools (e.g., GitHub Copilot and LLM assistants) were used to accelerate boilerplate coding, command crafting, and documentation wording.
- Model design, dataset logic, training/evaluation pipelines, experiments, and final decisions were implemented and verified by the author. All metrics/plots were generated from local runs.
- No datasets or model weights were shared with external AI services. External ideas are cited in the References; no verbatim third‑party code is included without attribution.
