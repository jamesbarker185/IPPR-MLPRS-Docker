from pydantic import BaseModel, Field
from typing import List, Optional, Dict

class BoundingBox(BaseModel):
    x: int
    y: int
    width: int
    height: int

class LicensePlate(BaseModel):
    plate_text: str = Field(..., description="Detected license plate text")
    confidence: float = Field(..., description="Confidence score of the detection")
    state: str = Field(..., description="Identified Malaysian state/region")
    bounding_box: BoundingBox = Field(..., description="Coordinates of the plate")
    area: float = Field(..., description="Area of the plate in pixels")
    # We might add method/phase info if needed for debugging

class PerformanceLogs(BaseModel):
    """Detailed performance timing for optimization experiments"""
    preprocessing_time_ms: float = Field(..., description="Time spent in image preprocessing pipeline (9 phases)")
    ocr_time_ms: float = Field(..., description="Time spent in OCR verification pipeline")
    total_time_ms: float = Field(..., description="Total processing time")

class DetectionResponse(BaseModel):
    success: bool
    processing_time_ms: float
    detections: List[LicensePlate]
    phases: Optional[Dict[str, str]] = Field(None, description="Base64 encoded images of processing phases (if requested)")
    candidates_analyzed: int
    ocr_engine: str
    performance_logs: Optional[PerformanceLogs] = Field(None, description="Detailed performance timing (if requested)")

class ImageDetectionResult(BaseModel):
    """Result for a single image in batch processing"""
    filename: str = Field(..., description="Original filename of the image")
    success: bool
    processing_time_ms: float
    detections: List[LicensePlate]
    candidates_analyzed: Optional[int] = None
    error: Optional[str] = Field(None, description="Error message if processing failed")
    performance_logs: Optional[PerformanceLogs] = Field(None, description="Detailed performance timing (if requested)")

class BatchDetectionResponse(BaseModel):
    """Response for batch image processing"""
    total_images: int
    successful: int
    failed: int
    total_processing_time_ms: float
    results: List[ImageDetectionResult]
    ocr_engine: str
