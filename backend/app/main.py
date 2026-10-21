"""
backend/app/main.py

FastAPI application entrypoint for AquaVigil.
Exposes health endpoints, model status, and the image detection endpoint.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import DetectionResponse
from app.services.detection import FishDiseaseDetectionService

app = FastAPI(
    title="AquaVigil API",
    description="Automated Fish Disease Detection & Aquaculture Health Screening API",
    version="0.1.0",
)

# Enable CORS for React frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

detection_service = FishDiseaseDetectionService()


@app.get("/")
def root():
    return {
        "project": "AquaVigil",
        "description": "Automated Fish Disease Detection System",
        "status": "online",
        "model_loaded": detection_service.is_ready,
        "note": "Phase 1 scaffold active. Model training scheduled for Phase 2.",
    }


@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "model_ready": detection_service.is_ready,
        "weights_path": str(detection_service.weights_path),
    }


@app.post("/api/detect", response_model=DetectionResponse)
async def detect_fish_disease(file: UploadFile = File(...)):
    """
    Accepts an uploaded fish image and runs YOLOv8 disease symptom detection.
    Returns detected bounding boxes, class names, and confidence scores.
    """
    if not detection_service.is_ready:
        return DetectionResponse(
            status="standby",
            message=(
                "Model weights are not currently loaded. AquaVigil is in Phase 1 scaffolding. "
                "Train a model using 'python training/train.py' to enable live inference."
            ),
            model_loaded=False,
            detections=[],
        )

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a valid image.")

    content = await file.read()
    try:
        detections, latency_ms = detection_service.run_detection(content)
        return DetectionResponse(
            status="success",
            message=f"Processed image successfully ({len(detections)} detection(s)).",
            model_loaded=True,
            detections=detections,
            inference_time_ms=latency_ms,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
