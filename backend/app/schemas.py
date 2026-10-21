"""
backend/app/schemas.py

Pydantic data models for the AquaVigil API.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    x1: float = Field(..., description="Top-left x pixel coordinate")
    y1: float = Field(..., description="Top-left y pixel coordinate")
    x2: float = Field(..., description="Bottom-right x pixel coordinate")
    y2: float = Field(..., description="Bottom-right y pixel coordinate")


class DetectionItem(BaseModel):
    class_id: int = Field(..., description="Detected class integer ID")
    class_name: str = Field(..., description="Detected class name (e.g. fin_rot)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score [0.0 - 1.0]")
    box: BoundingBox


class DetectionResponse(BaseModel):
    status: str
    message: str
    model_loaded: bool
    detections: List[DetectionItem] = []
    disclaimer: str = (
        "AquaVigil is an AI screening tool for preliminary visual inspection. "
        "Predictions do not substitute professional aquatic veterinary diagnosis."
    )
    inference_time_ms: Optional[float] = None
