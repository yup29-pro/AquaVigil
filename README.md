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

## 6. Run the Demo

The demo combines a FastAPI backend with a mobile-first HTML/CSS/JS web interface served at `/`.

### Step 1: Start the Server
From the project root:
```powershell
.venv\Scripts\python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
```
Confirm `GET http://127.0.0.1:8000/health` returns `model_loaded: true`.

### Step 2: Allow Inbound Traffic through Windows Firewall (Optional, for Phone Access)
Run once in an Administrator PowerShell window:
```powershell
New-NetFirewallRule -DisplayName "AquaVigil Demo Port 8000" -Direction Inbound -LocalPort 8000 -Protocol TCP -Action Allow
```

### Step 3: Access via Phone on Same Wi-Fi or Hotspot
1. Find your machine's IP address:
   ```powershell
   ipconfig
   ```
2. Open a mobile browser on any device connected to the same Wi-Fi or mobile hotspot:
   ```text
   http://<YOUR_PC_IP>:8000
   ```
   (e.g., `http://10.5.147.73:8000` or `http://192.168.137.1:8000`)

---

## 7. Known Limitations

- **Preliminary Baseline Model**: The current weights (`models/best.pt`) are a preliminary 25-epoch baseline trained with YOLOv8n.
- **Spot-Check Performance**: In an 11-image spot check on validation images across 5 distinct sources:
  - `lymphocystis` and `object` were not detected.
  - `branchitis` images also produced secondary `keratitis` boxes.
  - `keratitis` and `Aeromoniasis` showed strong single-class symptom localization.
- **Screening Aid Only**: AquaVigil is an AI screening aid, **not** a veterinary diagnosis tool. It cannot replace professional laboratory diagnostics or clinical examination.
- **Restricted Taxonomy**: Healthy fish and rare disease classes were excluded in baseline v1.
- **Validation metrics**: TBD

---

## 8. Current Project Status
- **Current Phase**: **Phase 4 Baseline Demo Built and Tested**
- **Inference Engine**: FastAPI + YOLOv8n CPU/GPU Inference with Aquaculture Knowledge Base
- **Test Set Quarantine**: The test split remains held out and has NOT been evaluated.

