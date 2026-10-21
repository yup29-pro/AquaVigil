# AquaVigil: Product Requirements Document (PRD)

## 1. Project Information
- **Project Name**: AquaVigil
- **Project Title**: A Computer Vision Framework for Automated Fish Disease Detection in Aquaculture Systems
- **Domain**: Artificial Intelligence, Computer Vision, Aquaculture
- **Primary Objective**: Detect and localize visible fish disease symptoms using computer vision
- **Dataset**: Fish Disease Dataset from Roboflow Universe (~9,011 images)
- **Initial Model**: YOLOv8 Object Detection (Ultralytics + PyTorch)
- **Backend**: FastAPI (Python)
- **Frontend**: React + TypeScript
- **Image Processing**: OpenCV, NumPy

---

## 2. Problem Statement
Fish diseases spread rapidly in aquaculture ponds and recirculation systems, leading to severe mortality and economic loss. Traditional inspection relies on manual visual observation by specialists, making continuous or high-frequency monitoring impractical.

AquaVigil provides an automated screening pipeline that identifies visible lesions, ulcers, or fin degradation, estimates localized bounding regions, and highlights confidence levels to assist aquaculture managers and veterinarians.

---

## 3. Functional Requirements

### FR-01: Dataset Management
- Load downloaded Roboflow dataset files.
- Inspect images, annotations, class names, and `data.yaml` configuration.
- Validate image-label 1-to-1 correspondence.
- Detect missing labels, malformed annotations, empty files, and duplicates.
- Verify train, validation, and test split ratios.

### FR-02: Model Training
- Use Ultralytics YOLOv8 for initial object detection.
- Provide configurable hyperparameters via `configs/model_config.yaml` (image size, batch size, epochs, device).
- Save model checkpoints (`best.pt`, `last.pt`) and training logs.
- Support both CPU and CUDA-enabled GPU acceleration.
- Support resumption from saved checkpoints.

### FR-03: Model Evaluation
- Evaluate performance on held-out test data (never training data).
- Compute metrics: Precision, Recall, mAP@50, mAP@50–95, Confusion Matrix, and inference latency.
- Export metrics in standard JSON / CSV / visual chart formats.

### FR-04: Disease Detection & Inference
- Accept raw fish images in standard formats (`.jpg`, `.jpeg`, `.png`).
- Perform forward inference using trained weights.
- Filter detections with confidence and IoU thresholds.
- Draw bounding boxes, class labels, and confidence tags.
- Handle zero-detection images safely and gracefully.

### FR-05: Web Application Interface
- Upload fish images via browser drag-and-drop.
- View real-time processed image with overlaid bounding boxes.
- Inspect detected disease classes, percentage confidences, and summary stats.
- Display a prominent medical disclaimer stating predictions are screening aids and require professional confirmation.

### FR-06: Explainability & Safety
- Display bounding box coordinates and classification confidence.
- Clearly present confidence indicators to avoid false certainty.
- Explicitly emphasize that bounding boxes localize visual symptoms rather than diagnose full etiology.

---

## 4. Technical Stack & Architecture
- **Model**: YOLOv8 (Ultralytics)
- **Deep Learning**: PyTorch
- **Image Processing**: OpenCV, NumPy, Pillow
- **Backend API**: FastAPI, Uvicorn
- **Frontend**: React, TypeScript, Vite, Recharts
