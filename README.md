# AquaVigil: Automated Fish Disease Detection in Aquaculture Systems

AquaVigil is an AI-powered computer vision framework that detects and localizes visible fish disease symptoms from aquaculture images using YOLOv8 object detection, a FastAPI inference engine, and a React + TypeScript monitoring dashboard.

---

## 1. Problem Statement & Scope

Fish diseases can spread rapidly in aquaculture facilities, causing substantial economic loss and high mortality. Traditional diagnostic approaches rely on manual, expert visual observation, making high-frequency monitoring impractical.

AquaVigil provides continuous screening capabilities:
- **Detection & Localization**: Identifies visual lesions, fin degradation, or ulcers with bounding boxes.
- **Diagnostic Disclaimer**: AquaVigil is a decision-support screening tool, **not** a replacement for qualified veterinary diagnosis.

---

## 2. Project Architecture & Directory Structure

```
AquaVigil/
├── README.md                      # Project documentation and guide
├── .gitignore                     # Git exclusion rules
├── requirements.txt               # Python package dependencies
├── .env.example                   # Environment variable template
│
├── dataset/
│   ├── raw/                       # Extracted Roboflow dataset
│   ├── processed/                 # Verified splits (created only upon approval)
│   └── README.md                  # Dataset extraction and handling guide
│
├── configs/
│   └── model_config.yaml          # Model, training, and output configurations
│
├── scripts/
│   ├── inspect_dataset.py         # Inspects files, directories, data.yaml, and classes
│   ├── validate_dataset.py        # Validates YOLO coordinates, labels, and classes
│   └── visualize_annotations.py   # Overlays bounding boxes on images safely
│
├── training/
│   ├── train.py                   # YOLOv8 model training script
│   ├── evaluate.py                # Model evaluation (mAP, precision, recall)
│   └── predict.py                 # Single-image inference script
│
├── models/                        # Checkpoints (e.g., best.pt, last.pt)
├── runs/                          # Ultralytics training run artifacts
├── results/
│   ├── metrics/                   # Saved evaluation metrics (.json)
│   └── predictions/               # Visualized inference images
│
├── backend/
│   ├── app/
│   │   ├── main.py                # FastAPI routes & endpoints
│   │   ├── schemas.py             # Request & response data models
│   │   └── services/
│   │       └── detection.py       # YOLOv8 inference service
│   └── README.md
│
├── frontend/
│   └── README.md                  # React + TypeScript dashboard overview
│
└── docs/
    ├── PRD.md                     # Product Requirements Document
    └── dataset_notes.md           # Dataset provenance, classes, and split log
```

---

## 3. Environment Setup

### Step 1: Create a Python Virtual Environment
Open PowerShell / Command Prompt in the `AquaVigil` directory:
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### Step 2: Install PyTorch (GPU or CPU)
Select the appropriate command for your machine:
- **For CPU Only (Default):**
  ```powershell
  pip install torch torchvision
  ```
- **For NVIDIA CUDA-enabled GPU (e.g. CUDA 12.1):**
  ```powershell
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
  ```

### Step 3: Install Project Dependencies
```powershell
pip install -r requirements.txt
```

---

## 4. Dataset Setup & Verification Workflow

1. Download the fish disease dataset from Roboflow Universe (~9,011 images).
2. Extract the archive contents into `dataset/raw/`.
3. Verify that `dataset/raw/data.yaml` is present.
4. Run the inspection script:
   ```powershell
   python scripts/inspect_dataset.py --dataset-dir dataset/raw
   ```
5. Validate labels and check for formatting anomalies:
   ```powershell
   python scripts/validate_dataset.py --data-dir dataset/raw --check-duplicates
   ```
6. Preview sample annotations visually:
   ```powershell
   python scripts/visualize_annotations.py --image dataset/raw/train/images/<sample_image>.jpg
   ```

> **IMPORTANT SPLIT WARNING**:
> Check whether the dataset includes `valid` or `test` sets. If all images are placed in `train/` with 0 validation images, **do not proceed directly to training**. We must first establish an independent, vetted split to prevent data contamination and ensure valid evaluation.

---

## 5. Training, Evaluation, and Prediction (Phase 2)

### Train the Baseline Model
```powershell
python training/train.py --config configs/model_config.yaml --device cpu
```

### Evaluate the Model
```powershell
python training/evaluate.py --weights models/best.pt --split val
```

### Run Prediction on a Test Image
```powershell
python training/predict.py --weights models/best.pt --source sample_fish.jpg
```

---

## 6. Future Integration (Phases 3 & 4)

- **Backend (FastAPI)**: Serves detection endpoints at `/api/detect`.
  ```powershell
  uvicorn backend.app.main:app --reload
  ```
- **Frontend (React + TypeScript)**: Drag-and-drop fish image upload interface with interactive symptom localization and veterinary disclaimers.

---

## 7. Current Project Status
- **Current Phase**: **Phase 1 Complete (Scaffolding & Scripts)**
- **Next Phase**: **Phase 2 (Dataset Inspection, Verification, and Split Confirmation)**
- **Note**: Training has not been started. Awaiting dataset placement and inspection.
