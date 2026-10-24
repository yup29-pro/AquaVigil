"""
backend/app/main.py

FastAPI application entrypoint for AquaVigil.
Exposes:
- GET  /health   -> Model loading status and metadata
- POST /predict  -> Image screening, bounding boxes, annotated image base64, latency, disclaimer
- Static files mounted at root / from frontend/static/
"""

import base64
import io
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, List, Dict, Any

import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from ultralytics import YOLO

from backend.app.knowledge import KNOWLEDGE_BASE

# Project root resolution
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
WEIGHTS_PATH = PROJECT_ROOT / "models" / "best.pt"
STATIC_DIR = PROJECT_ROOT / "frontend" / "static"

# Global state for loaded model
model_instance: Optional[YOLO] = None
model_load_error: Optional[str] = None

STANDARD_DISCLAIMER = (
    "AquaVigil is an AI screening aid only. Not a veterinary diagnosis. "
    "Do not initiate pharmaceutical treatments based solely on automated screening. "
    "Confirm any concerns with a certified aquaculture veterinarian or aquatic animal health professional."
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_instance, model_load_error
    print(f"[*] AquaVigil initializing... Checking weights at: {WEIGHTS_PATH}")
    if WEIGHTS_PATH.exists():
        try:
            # Explicitly load model on CPU
            model_instance = YOLO(str(WEIGHTS_PATH))
            print(f"[+] Model loaded successfully on CPU. Classes: {model_instance.names}")
        except Exception as e:
            model_load_error = str(e)
            print(f"[!] Failed to load model: {e}")
    else:
        model_load_error = f"Model weights not found at {WEIGHTS_PATH}"
        print(f"[!] {model_load_error}")
    yield


app = FastAPI(
    title="AquaVigil API",
    description="Automated Fish Disease Detection & Aquaculture Health Screening Demo API",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for external devices on local network / hotspot
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    is_loaded = model_instance is not None
    classes = model_instance.names if is_loaded else {}
    return {
        "status": "healthy",
        "model_loaded": is_loaded,
        "weights_path": str(WEIGHTS_PATH),
        "weights_exists": WEIGHTS_PATH.exists(),
        "classes": classes,
        "error": model_load_error,
        "device": "cpu",
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    conf: float = Form(0.25),
):
    if model_instance is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model not loaded. Error: {model_load_error or 'Weights missing'}",
        )

    # Validate uploaded content type
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Please upload a valid image file (JPEG, PNG, WEBP).",
        )

    image_bytes = await file.read()
    if not image_bytes or len(image_bytes) < 16:
        raise HTTPException(
            status_code=400,
            detail="The uploaded image file is empty or corrupted.",
        )

    try:
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Unable to decode image file: {str(e)}",
        )

    # Clamp confidence threshold
    conf_threshold = max(0.01, min(0.99, float(conf)))

    # Run inference on CPU
    start_time = time.perf_counter()
    results = model_instance.predict(
        source=pil_image,
        conf=conf_threshold,
        device="cpu",
        verbose=False,
    )
    inference_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

    detections: List[Dict[str, Any]] = []
    annotated_base64: str = ""

    if results and len(results) > 0:
        res = results[0]

        # Extract plotted image with bounding boxes
        plot_bgr = res.plot()  # numpy array BGR
        plot_rgb = cv2.cvtColor(plot_bgr, cv2.COLOR_BGR2RGB)
        plot_pil = Image.fromarray(plot_rgb)

        buffered = io.BytesIO()
        plot_pil.save(buffered, format="JPEG", quality=88)
        annotated_base64 = "data:image/jpeg;base64," + base64.b64encode(buffered.getvalue()).decode("utf-8")

        # Parse boxes
        boxes = res.boxes
        if boxes is not None and len(boxes) > 0:
            for i in range(len(boxes)):
                box = boxes[i]
                cls_id = int(box.cls[0].item())
                cls_name = model_instance.names.get(cls_id, str(cls_id))
                score = round(float(box.conf[0].item()), 4)
                xyxy = [round(float(coord), 2) for coord in box.xyxy[0].tolist()]

                kb_info = KNOWLEDGE_BASE.get(cls_name, {
                    "name": cls_name,
                    "is_disease": True,
                    "description": "Information not available for this class.",
                    "visible_signs": [],
                    "general_guidance": ["Consult an aquaculture health professional."],
                })

                detections.append({
                    "class_id": cls_id,
                    "class": cls_name,
                    "confidence": score,
                    "confidence_pct": round(score * 100.0, 1),
                    "box": {
                        "x1": xyxy[0],
                        "y1": xyxy[1],
                        "x2": xyxy[2],
                        "y2": xyxy[3],
                    },
                    "knowledge": kb_info,
                })

    # If zero detections, return original image as preview
    if not annotated_base64:
        buffered = io.BytesIO()
        pil_image.save(buffered, format="JPEG", quality=88)
        annotated_base64 = "data:image/jpeg;base64," + base64.b64encode(buffered.getvalue()).decode("utf-8")

    return {
        "status": "success",
        "count": len(detections),
        "detections": detections,
        "annotated_image": annotated_base64,
        "inference_ms": inference_ms,
        "confidence_threshold": conf_threshold,
        "disclaimer": STANDARD_DISCLAIMER,
    }


# Ensure static directory exists
STATIC_DIR.mkdir(parents=True, exist_ok=True)

# Mount static folder for frontend demo at root
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
