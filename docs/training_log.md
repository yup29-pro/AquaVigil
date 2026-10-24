# AquaVigil Baseline Training Log

## Run Details
- **Environment**: Google Colab (NVIDIA T4 GPU)
- **Base Architecture**: `yolov8n.pt`
- **Dataset**: `baseline_v1`
  - Split counts: Train: 5,693 | Valid: 396 | Test: 398
  - Target classes (5):
    - `0: Aeromoniasis`
    - `1: branchitis`
    - `2: keratitis`
    - `3: lymphocystis`
    - `4: object`

## Training Hyperparameters
- **Resolution (`imgsz`)**: 640
- **Epochs**: 25
- **Batch Size**: 16
- **Early Stopping Patience**: 15
- **Random Seed**: 42
- **Dataloader Workers**: 2
- **Checkpoint Save Period**: 5 epochs

## Split Discipline & Test Split Protocol
- All model training and validation checks were conducted strictly on the **train** and **valid** partitions.
- **The test split has NOT been used or evaluated.** In accordance with the AquaVigil evaluation protocol, the test split remains strictly quarantined until final benchmarking.
