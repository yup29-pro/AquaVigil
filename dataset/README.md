# AquaVigil Dataset Directory

This directory is reserved for the fish disease dataset from Roboflow Universe.

## Recommended Directory Structure

```
AquaVigil/
└── dataset/
    ├── raw/              # Direct extracted archive from Roboflow Universe
    │   ├── data.yaml     # Dataset configuration (classes, paths)
    │   ├── README.dataset.txt (or similar if provided by Roboflow)
    │   ├── train/
    │   │   ├── images/
    │   │   └── labels/
    │   ├── valid/        # (if present in dataset)
    │   │   ├── images/
    │   │   └── labels/
    │   └── test/         # (if present in dataset)
    │       ├── images/
    │       └── labels/
    └── processed/        # Verified splits or derived subsets (created only upon approval)
```

## How to Place and Extract Your Dataset

1. Place the downloaded `.zip` archive into the `dataset/` root or directly extract it into `dataset/raw/`.
2. Ensure the resulting structure contains the dataset configuration file (typically named `data.yaml`) and the image/label folders.
3. If the archive extracts into a subfolder (e.g., `dataset/raw/Fish-Disease-Detection-1/`), move the contents directly to `dataset/raw/` so `dataset/raw/data.yaml` is accessible at root level.
4. Run the inspection script:
   ```bash
   python scripts/inspect_dataset.py --dataset-dir dataset/raw
   ```

## Split Integrity & Known Warning

> **CRITICAL DATASET NOTICE**:
> Previous versions of this Roboflow dataset have been observed to bundle all images (~9,011 images) into the `train` set with 0 validation and 0 test images.
>
> - Never evaluate models on training images alone.
> - Inspect the splits using `scripts/inspect_dataset.py` before training.
> - Do not automatically split or move files. If the dataset lacks validation/test splits, we will review the data distribution, inspect for augmented/duplicate images, and create a vetted split in `dataset/processed/`.
