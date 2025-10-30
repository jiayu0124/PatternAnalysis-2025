# Prostate 3D Segmentation with UNet3D (COMP3710 Project 7)

Author: 48300494  
Repository: PatternAnalysis-2025 (topic-recognition)

Abstract
This report documents a complete solution to Project 7: Segment the (downsampled) Prostate 3D dataset using a 3D UNet with all labels having Dice ≥ 0.7 on the test set. We implement a compact 3D UNet in PyTorch, a robust training/evaluation pipeline, and demonstrate reproducible results exceeding the required threshold on a held-out test split. We also include design choices, training curves, per-class metrics, and practical tips for reproducibility and troubleshooting.

Table of Contents
- 1. Problem & Objective
- 2. Dataset & Splits
- 3. Method
  - 3.1 Model Architecture (UNet3D)
  - 3.2 Loss & Metrics
  - 3.3 Optimizer & Training Strategy
- 4. Implementation Overview
  - 4.1 Code Structure
  - 4.2 Key Design Decisions
- 5. Experiments
  - 5.1 Training Setup
  - 5.2 Validation Dynamics (Curves)
  - 5.3 Test Set Results (Main)
  - 5.4 Short-run Experiments (Ablations)
- 6. Error Analysis & Discussion
- 7. Reproducibility
  - 7.1 Environment & Dependencies
  - 7.2 Data Preparation
  - 7.3 Training Commands
  - 7.4 Evaluation Commands
- 8. Troubleshooting & Tips
- 9. Commit Log & PR Checklist
- 10. Limitations & Future Work
- 11. References
- 12. AI Assistance Declaration

1. Problem & Objective
We address Project 7 (Normal Difficulty – 3D UNet): Segment the Prostate 3D dataset into multiple anatomical classes using a 3D UNet, achieving Dice ≥ 0.7 for every class on a held-out test set. The target is a robust, reproducible pipeline with documented instructions, code, and results.

2. Dataset & Splits
- Source: HipMRI Study (Prostate radiotherapy MR volumes) — downsampled 3D data. Labels include: background and multiple structures (e.g., body outline, bone, bladder, rectum, prostate). In this implementation we consider 6 classes (0–5).
- Format: NIfTI (.nii.gz) volumes organized into:
  - recognition/prostate3d_unet_48300494/data/semantic_MRs_anon/
  - recognition/prostate3d_unet_48300494/data/semantic_labels_anon/
- Splits: Plain-text lists provided under recognition/prostate3d_unet_48300494/splits/
  - train_list.txt
  - val_list.txt
  - test_list.txt
- We justify 70/15/15 (train/val/test) stratified by patient IDs to avoid leakage. Validation curves inform early/late training behavior; test set is used only for final reporting.

3. Method
3.1 Model Architecture (UNet3D)
- Input: 1×D×H×W (single-channel MR volume)
- Base channels: 16
- Encoder (down path): 4 levels of Conv3d-BN-ReLU blocks with stride-2 downsampling (via max-pool or conv), doubling channels at each level: 16→32→64→128→256.
- Bottleneck: Conv3d-BN-ReLU×2 at the deepest resolution.
- Decoder (up path): Transposed conv upsampling; skip-connections concatenate encoder features; Conv3d-BN-ReLU×2 to refine.
- Output: 1×C×D×H×W logits (C=6 classes). Softmax applied in loss/metrics.
- Rationale: A compact UNet (base 16) balances memory and convergence; suitable for desktop GPUs while preserving necessary 3D context.

3.2 Loss & Metrics
- Loss: Soft Dice loss on softmax probabilities vs. one-hot labels (see utils.py: DiceLoss). Default run uses Dice-only (weight=1.0); CE can be blended if desired.
- Metric: Per-class Dice coefficients on validation/test. We report mean Dice and per-class breakdown, and indicate PASS if all classes ≥ threshold.

3.3 Optimizer & Training Strategy
- Optimizer: Adam, lr=1e-3.
- Mixed Precision: AMP enabled when CUDA is available for speed/memory efficiency.
- Data Loading: Custom pad-collate to batch variable-sized volumes; pin_memory + non_blocking H2D copies; cuDNN benchmark for stable shapes.
- Augmentation: Random axis flips with 50% probability.

4. Implementation Overview
4.1 Code Structure
- recognition/prostate3d_unet_48300494/
  - modules.py: UNet3D definition
  - dataset.py: Prostate3DDataset (NIfTI IO, normalization, augmentation)
  - utils.py: DiceLoss and per-class Dice computation
  - train.py: Training loop, validation, curves, logging, AMP, best checkpoint saving
  - predict.py: Evaluation on a list of cases; optional save predictions
  - splits/: train/val/test lists
  - outputs/: best_model.pt, curves.png, metrics.csv, metrics_per_class.csv, logs

4.2 Key Design Decisions
- Compact UNet with base_channels=16 to ensure stable training on consumer GPU.
- Dice-centric objective for multi-class segmentation with class imbalance.
- Log per-class Dice and write CSV for traceability; save best model by val mean Dice.

5. Experiments
5.1 Training Setup
- Hardware: Desktop GPU (CUDA 12.x), AMP on.
- Hyperparameters: epochs=50, batch_size=1 (3D volumes), lr=1e-3, base_channels=16.
- Validation frequency: Every epoch; curves saved to outputs/curves.png.

5.2 Validation Dynamics (Curves)
- Training and validation losses decrease smoothly; mean Val Dice increases gradually.
- In early epochs small structures (e.g., prostate/rectum) may lag. Around epoch ~16 all classes exceed 0.7 (first PASS), stabilizing thereafter (see train_full.log; curves.png).

5.3 Test Set Results (Main)
- Using the best checkpoint (best_model.pt) from the 50-epoch run, we evaluate on the test split:
  - Per-class Dice:
    - Class 0: 0.9922
    - Class 1: 0.9783
    - Class 2: 0.9050
    - Class 3: 0.9268
    - Class 4: 0.8490
    - Class 5: 0.8340
  - Mean Dice: 0.9142
  - PASS: All classes ≥ 0.7 (meets Project 7 requirement).

5.4 Short-run Experiments (Ablations)
- We explored short-run (≤15 epochs) configurations for faster iteration:
  - Background handling (ignore vs. include in loss / threshold) and mild class weighting.
  - Short runs are informative but not a guarantee of PASS; full training yields stable PASS from epoch ~16 onwards.
- Final reported results use the 50-epoch configuration without experimental toggles.

6. Error Analysis & Discussion
- Small-structure classes (e.g., prostate) are harder early on; Dice can be near-zero in the first few epochs until the model learns fine details.
- The compact model converges stably; increasing base_channels can improve early dynamics at the cost of memory.
- Dice vs. CE trade-off: Pure Dice worked well here; adding CE is an option for further stabilization.

7. Reproducibility
7.1 Environment & Dependencies
- Python 3.12; PyTorch with CUDA optional; nibabel, numpy, matplotlib.
- Suggested installation (adapt per your CUDA):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install nibabel numpy matplotlib
```

7.2 Data Preparation
- Place NIfTI volumes in:
  - recognition/prostate3d_unet_48300494/data/semantic_MRs_anon/
  - recognition/prostate3d_unet_48300494/data/semantic_labels_anon/
- Ensure splits exist under recognition/prostate3d_unet_48300494/splits/ with case IDs.

7.3 Training Commands (Windows CMD)
- Full training (50 epochs):
```cmd
python -u recognition\prostate3d_unet_48300494\train.py ^
  --root_img recognition\prostate3d_unet_48300494\data\semantic_MRs_anon ^
  --root_lbl recognition\prostate3d_unet_48300494\data\semantic_labels_anon ^
  --train_list recognition\prostate3d_unet_48300494\splits\train_list.txt ^
  --val_list recognition\prostate3d_unet_48300494\splits\val_list.txt ^
  --out_dir recognition\prostate3d_unet_48300494\outputs ^
  --epochs 50 ^
  --base_channels 16 ^
  --num_workers 0
```

7.4 Evaluation Commands (Windows CMD)
- Test set evaluation (reported above):
```cmd
python -u recognition\prostate3d_unet_48300494\predict.py ^
  --root_img recognition\prostate3d_unet_48300494\data\semantic_MRs_anon ^
  --root_lbl recognition\prostate3d_unet_48300494\data\semantic_labels_anon ^
  --test_list recognition\prostate3d_unet_48300494\splits\test_list.txt ^
  --model recognition\prostate3d_unet_48300494\outputs\best_model.pt ^
  --num_classes 6 ^
  --base_channels 16
```
- Optional save predictions as NIfTI:
```cmd
python -u recognition\prostate3d_unet_48300494\predict.py ... ^
  --save_dir recognition\prostate3d_unet_48300494\outputs\preds
```

8. Troubleshooting & Tips
- Mismatched base_channels during evaluation → ensure `--base_channels` matches the training checkpoint (here 16).
- Slow first epoch → 3D IO and cuDNN autotune can make the first batches slow; later epochs speed up.
- CUDA OOM → reduce base_channels (e.g., 12/16), or use CPU with no_cuda.
- Data not found → verify case IDs in splits match filenames (dataset.py has flexible matching via glob).

9. Commit Log & PR Checklist
- Commit Log: Progressive commits exist (dataset, model, train, metrics, curves, evaluation, docs). Avoid single-commit submissions.
- PR Checklist:
  - [x] Code compiles and runs; best_model.pt produced; tests documented
  - [x] README has usage, environment, results (with per-class Dice ≥ 0.7 on test)
  - [x] No datasets/models committed; .gitignore excludes data/outputs
  - [x] Branch: topic-recognition; PR into course repo topic-recognition
  - [x] Include brief design/experiments notes and figures (curves.png)

10. Limitations & Future Work
- Capacity vs. efficiency: Larger base channels may further improve small-structure Dice.
- Augmentations are minimal; adding intensity/elastic aug could help generalization.
- Hybrid Dice+CE or focal terms can stabilize minority-class learning.

11. References
- Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. In MICCAI 2015, LNCS 9351, pp. 234–241. Springer. doi:10.1007/978-3-319-24574-4_28. Preprint: https://arxiv.org/abs/1505.04597
- Çiçek, Ö., Abdulkadir, A., Lienkamp, S.S., Brox, T., & Ronneberger, O. (2016). 3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation. In MICCAI 2016, LNCS 9901, pp. 424–432. Springer. doi:10.1007/978-3-319-46723-8_49. Preprint: https://arxiv.org/abs/1606.06650
- Isensee, F., Jaeger, P.F., Kohl, S.A.A., Petersen, J., & Maier-Hein, K.H. (2021). nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. Nature Methods, 18, 203–211. doi:10.1038/s41592-020-01008-z. Preprint: https://arxiv.org/abs/1809.10486
- COMP3710 Assessment Brief (Project 7): Course-provided specification for Pattern Analysis recognition tasks (accessed Oct 2025).

12. AI Assistance Declaration
- Policy: In accordance with the course brief (Section 1.5 Use of Artificial Intelligence), AI tooling was permitted to assist learning and workflows.
- Tools used: GitHub Copilot (IDE suggestions and completions) and comparable LLM assistants for drafting small code snippets, refactoring, PowerShell/CMD command crafting, and documentation wording.
- Scope of assistance: AI suggestions were accepted and adapted primarily for boilerplate, logging, plotting, and documentation. The model design choices, dataset logic, training/evaluation scripts, experiments, hyperparameters, and final implementation decisions were made by the author. All code paths were reviewed and executed locally; metrics and plots were produced from local runs.
- Originality and attribution: No verbatim third‑party code was pasted without attribution. Where external ideas influenced the implementation (e.g., UNet3D conventions), appropriate references are included in the References section. Commit history reflects progressive, individual development.
- Data & privacy: No confidential datasets or patient information were uploaded to external AI services. Only generic code fragments or commands were used with AI tools.
- Validation: AI outputs can be incorrect; therefore, all generated or suggested code was tested and, where necessary, rewritten. Reported test‑set metrics were reproduced locally with the provided scripts.
