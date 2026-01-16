import cv2
import numpy as np
import time
import logging
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from typing import Optional, List
from contextlib import asynccontextmanager
import io
from PIL import Image

from .pipeline import process_image_through_phases
from .ocr import load_ocr_model, ocr_verification_pipeline, extract_plate_text
from .utils import identify_state, enhance_plate_region, encode_image_base64
from .models import DetectionResponse, LicensePlate, BoundingBox, ImageDetectionResult, BatchDetectionResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
ocr_model = None
ocr_engine_name = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global ocr_model, ocr_engine_name
    logger.info("Loading OCR model...")
    ocr_model, ocr_engine_name = load_ocr_model()
    if ocr_model:
        logger.info(f"OCR Engine {ocr_engine_name} loaded successfully")
    else:
        logger.error("Failed to load OCR Engine")
    yield
    # Shutdown
    logger.info("Shutting down...")

app = FastAPI(title="Malaysian LPR Microservice", lifespan=lifespan)

def load_image_from_bytes(data: bytes) -> np.ndarray:
    try:
        image = Image.open(io.BytesIO(data))
        # Convert to RGB (OpenCV uses BGR, but pipeline handles RGB conversion internally if needed,
        # usually standard is to read as RGB with PIL, then convert to BGR for OpenCV usage or keep as RGB.
        # The pipeline expects RGB or BGR? 
        # Source app: img = Image.open(uploaded_file); img_np = np.array(img) -> This is RGB (PIL default).
        # Pipeline: if len(img_np.shape) == 3: gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        # So pipeline expects RGB.
        
        # However, OpenCV internal operations usually expect BGR if we use cv2.imread.
        # But PIL.Image.open() returns RGB. np.array(pil_img) is RGB.
        # The pipeline code converts using COLOR_RGB2GRAY, so it assumes RGB input.
        # That is correct.
        
        # Handle alpha channel
        if image.mode == 'RGBA':
            image = image.convert('RGB')
            
        return np.array(image)
    except Exception as e:
        logger.error(f"Image loading failed: {e}")
        raise HTTPException(status_code=400, detail="Invalid image file")

@app.post("/detect", response_model=DetectionResponse)
async def detect_license_plate(
    file: UploadFile = File(...),
    return_phases: bool = Form(False)
):
    start_time = time.time()
    
    if not ocr_model:
        raise HTTPException(status_code=503, detail="OCR model not initialized")

    # Read image
    contents = await file.read()
    img_np = load_image_from_bytes(contents)
    
    # Run pipeline
    try:
        phases, candidates = process_image_through_phases(img_np)
        
        # OCR Verification
        verified_candidates = ocr_verification_pipeline(ocr_model, ocr_engine_name, candidates, phases)
        
        detections = []
        # Filter for candidates with text or valid text
        # Logic similar to app.py: we usually take the best one, but here we can return all valid candidates
        
        for cand in verified_candidates:
            x, y, w, h, boosted_area, text, conf = cand
            
            if text: # Only return if text was found
                state = identify_state(text)
                
                detection = LicensePlate(
                    plate_text=text,
                    confidence=conf,
                    state=state,
                    bounding_box=BoundingBox(x=x, y=y, width=w, height=h),
                    area=float(boosted_area)
                )
                detections.append(detection)
        
        # Prepare phases if requested
        phase_images = {}
        if return_phases:
            for name, img in phases.items():
                if name == 'original': continue # Skip original to save bandwidth? Or keep it?
                # Ensure image is compatible with imencode
                # Some phases might be float or boolean, need convert
                if len(img.shape) == 2: # Grayscale
                    phase_images[name] = encode_image_base64(img)
                else: # RGB/BGR
                    # Pipeline uses internal OpenCV which is BGR usually? 
                    # Wait, input was RGB. Pipeline converts to Gray. 
                    # Visualization results (detection_result) is drawn on copy of RGB input.
                    # cv2.rectangle uses BGR color (255,0,0) is Blue.
                    # If input is RGB, (255,0,0) is Red.
                    # Streamlit st.image handles RGB by default.
                    # cv2 operations (rectangle) modify the array in place. 
                    # If we passed RGB (PIL), and used cv2.rectangle with (255,0,0) -> Red rectangle.
                    # So 'detection_result' is RGB with drawn rectangles.
                    # But encoding expects BGR for standard colors? 
                    # cv2.imencode expects BGR. 
                    # So if 'detection_result' is RGB, we should convert to BGR before encode.
                    
                    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    phase_images[name] = encode_image_base64(img_bgr)

        processing_time = (time.time() - start_time) * 1000
        
        return DetectionResponse(
            success=True,
            processing_time_ms=processing_time,
            detections=detections,
            phases=phase_images if return_phases else None,
            candidates_analyzed=len(candidates),
            ocr_engine=ocr_engine_name
        )

    except Exception as e:
        logger.exception("Processing failed")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect/batch", response_model=BatchDetectionResponse)
async def detect_license_plates_batch(
    files: List[UploadFile] = File(..., description="Multiple image files to process"),
    return_phases: bool = Form(False)
):
    """
    Process multiple images in a single request.
    Returns individual results for each image.
    """
    start_time = time.time()
    
    if not ocr_model:
        raise HTTPException(status_code=503, detail="OCR model not initialized")
    
    results = []
    successful = 0
    failed = 0
    
    for file in files:
        image_start = time.time()
        try:
            # Read and process image
            contents = await file.read()
            img_np = load_image_from_bytes(contents)
            
            # Run pipeline
            phases, candidates = process_image_through_phases(img_np)
            verified_candidates = ocr_verification_pipeline(ocr_model, ocr_engine_name, candidates, phases)
            
            # Extract detections
            detections = []
            for cand in verified_candidates:
                x, y, w, h, boosted_area, text, conf = cand
                if text:
                    state = identify_state(text)
                    detection = LicensePlate(
                        plate_text=text,
                        confidence=conf,
                        state=state,
                        bounding_box=BoundingBox(x=x, y=y, width=w, height=h),
                        area=float(boosted_area)
                    )
                    detections.append(detection)
            
            processing_time = (time.time() - image_start) * 1000
            
            results.append(ImageDetectionResult(
                filename=file.filename,
                success=True,
                processing_time_ms=processing_time,
                detections=detections,
                candidates_analyzed=len(candidates),
                error=None
            ))
            successful += 1
            
        except Exception as e:
            processing_time = (time.time() - image_start) * 1000
            logger.exception(f"Failed to process {file.filename}")
            
            results.append(ImageDetectionResult(
                filename=file.filename,
                success=False,
                processing_time_ms=processing_time,
                detections=[],
                candidates_analyzed=None,
                error=str(e)
            ))
            failed += 1
    
    total_time = (time.time() - start_time) * 1000
    
    return BatchDetectionResponse(
        total_images=len(files),
        successful=successful,
        failed=failed,
        total_processing_time_ms=total_time,
        results=results,
        ocr_engine=ocr_engine_name
    )

@app.get("/health")
def health_check():
    return {"status": "ok", "ocr_ready": ocr_model is not None}

@app.get("/info")
def system_info():
    return {
        "service": "Malaysian LPR Microservice",
        "version": "1.0.0",
        "ocr_engine": ocr_engine_name,
        "phases": 9
    }
