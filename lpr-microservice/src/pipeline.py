import cv2
import numpy as np
import pywt
import logging
from typing import Tuple, Dict, List

from .detection import detect_license_plate_regions

logger = logging.getLogger(__name__)

def process_image_through_phases(img_np: np.ndarray) -> Tuple[Dict[str, np.ndarray], List[Tuple[int, int, int, int, float]]]:
    """
    Process image through all 9 phases and return processed images and plate candidates
    
    Args:
        img_np: Input image as numpy array
        
    Returns:
        Tuple of (phases_dict, plate_candidates)
    """
    phases = {}
    
    # Convert to grayscale for processing
    if len(img_np.shape) == 3:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_np.copy()
        
    phases['original'] = img_np
    phases['grayscale'] = gray
    
    # Phase 2: Image Enhancement
    enhanced = cv2.equalizeHist(gray)
    gamma = 1.2
    gamma_corrected = np.array(255 * (enhanced / 255) ** gamma, dtype='uint8')
    phases['enhanced'] = gamma_corrected
    
    # Phase 3: Image Restoration
    restored = cv2.bilateralFilter(gamma_corrected, 11, 17, 17)
    phases['restored'] = restored
    
    # Phase 4: Color Image Processing
    if len(img_np.shape) == 3:
        hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
        value_channel = hsv[:, :, 2]
    else:
        value_channel = gray.copy()
    phases['color_processed'] = value_channel
    
    # Phase 5: Wavelet Transform
    try:
        coeffs2 = pywt.dwt2(restored, 'db4')
        LL, (LH, HL, HH) = coeffs2
        detail_combined = np.sqrt(LH**2 + HL**2 + HH**2)
        
        # Normalize to prevent division by zero
        detail_min, detail_max = np.min(detail_combined), np.max(detail_combined)
        if detail_max > detail_min:
            detail_norm = np.uint8(255 * (detail_combined - detail_min) / (detail_max - detail_min))
        else:
            detail_norm = np.zeros_like(detail_combined, dtype=np.uint8)
    except Exception as e:
        logger.warning(f"Wavelet transform failed: {e}")
        detail_norm = restored.copy()
    
    phases['wavelet'] = detail_norm
    
    # Phase 6: Image Compression
    h_orig, w_orig = restored.shape
    # Ensure minimum size to prevent errors
    new_w, new_h = max(1, w_orig//4), max(1, h_orig//4)
    compressed = cv2.resize(restored, (new_w, new_h))
    decompressed = cv2.resize(compressed, (w_orig, h_orig))
    phases['compressed'] = decompressed
    
    # Phase 7: Morphological Processing
    kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph_close = cv2.morphologyEx(restored, cv2.MORPH_CLOSE, kernel_rect)
    morph_grad = cv2.morphologyEx(morph_close, cv2.MORPH_GRADIENT, kernel_rect)
    phases['morphological'] = morph_grad
    
    # Phase 8: Segmentation
    adaptive_thresh = cv2.adaptiveThreshold(
        restored, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    phases['segmented'] = adaptive_thresh
    
    # Phase 9: Representation & Description
    # Try detection on multiple phases to find best results
    candidates_enhanced = detect_license_plate_regions(enhanced)
    candidates_restored = detect_license_plate_regions(restored) 
    candidates_morph = detect_license_plate_regions(morph_grad)
    
    # Combine all candidates and remove duplicates
    all_phase_candidates = []
    for candidates in [candidates_enhanced, candidates_restored, candidates_morph]:
        all_phase_candidates.extend(candidates)
    
    # Remove duplicates (similar positions)
    unique_candidates = []
    for candidate in all_phase_candidates:
        x, y, w, h, area = candidate
        is_duplicate = False
        for existing in unique_candidates:
            ex, ey, ew, eh, ea = existing
            # Check if centers are close (within 30 pixels)
            if abs((x + w/2) - (ex + ew/2)) < 30 and abs((y + h/2) - (ey + eh/2)) < 30:
                is_duplicate = True
                # Keep the larger area
                if area > ea:
                    unique_candidates.remove(existing)
                    unique_candidates.append(candidate)
                break
        if not is_duplicate:
            unique_candidates.append(candidate)
    
    # Sort by intelligent scoring for license plates
    def license_plate_score(candidate):
        if len(candidate) == 6:
            x, y, w, h, area, method = candidate
        else:
            x, y, w, h, area = candidate
            method = "unknown"
        
        # Get image dimensions for position scoring
        img_height, img_width = enhanced.shape[:2]
        
        # Base score from area
        area_score = area
        
        # Smart edge penalty (less harsh for small motorcycle plates)
        edge_penalty = 1.0
        
        # Check if this is likely a small motorcycle plate (area-based detection)
        is_small_plate = area < 1000  # Likely motorcycle if area < 1000
        
        if (y <= 2 or x <= 2 or  # Only penalize VERY close to edges
            (y + h) >= (img_height - 2) or (x + w) >= (img_width - 2)):  # Near opposite edges
            edge_penalty = 0.001  # Extreme penalty for image border detections
        elif not is_small_plate and (y < 20 or x < 20 or  # Regular edge penalty only for large plates
            (y + h) > (img_height - 20) or (x + w) > (img_width - 20)):  # Regular far edges
            edge_penalty = 0.1  # Moderate penalty for large plates near edges
        elif is_small_plate and (y < 10 or x < 10 or  # More lenient for small plates
            (y + h) > (img_height - 10) or (x + w) > (img_width - 10)):  # More lenient for small plates
            edge_penalty = 0.5  # Light penalty for small plates near edges
        
        # Prefer license plates in lower 2/3 of image (where cars are)
        position_score = 1.0
        center_y = y + h/2
        if center_y > img_height * 0.3:  # Lower 70% of image
            position_score = 2.0  # Strong bonus for car area
        if center_y > img_height * 0.7:  # Bottom 30% of image
            position_score = 1.5  # Still good but slightly lower
        
        # MOTORCYCLE-OPTIMIZED size scoring (prioritizes small plates)
        size_score = 1.0
        if area > 25000:  # Very large areas are likely false positives
            size_score = 0.2
        elif area > 15000:  # Large areas are suspicious
            size_score = 0.4
        elif 5000 <= area <= 15000:  # Good size range for car license plates
            size_score = 1.5
        elif 2000 <= area <= 5000:  # Smaller plates (cars and large motorcycles)
            size_score = 1.8  # Higher bonus for smaller realistic plates
        elif 800 <= area <= 2000:  # MOTORCYCLE SIZE RANGE - very high priority
            size_score = 2.5   # INCREASED bonus for motorcycle-sized plates
        elif 400 <= area <= 800:   # Small motorcycles - very high priority
            size_score = 2.8   # HIGHEST bonus for small plates
        elif 200 <= area <= 400:   # Tiny distant motorcycles - high priority
            size_score = 2.2   # HIGH bonus for tiny plates
        elif 100 <= area <= 200:   # Micro distant motorcycles - still valid
            size_score = 1.8   # Good bonus for micro plates
        elif 50 <= area <= 100:    # Very tiny but possible distant plates
            size_score = 1.2   # Modest bonus
        else:
            size_score = 0.3   # Very small or very large
        
        # IMPROVED aspect ratio preference with motorcycle support
        aspect_ratio = w / h
        aspect_score = 1.0
        if 3.0 <= aspect_ratio <= 4.5:  # Ideal single-line car plates
            aspect_score = 1.2
        elif 1.4 <= aspect_ratio <= 2.2:  # MOTORCYCLE ASPECT RATIO - high priority
            aspect_score = 1.5  # INCREASED bonus for motorcycle aspect ratios
        elif 1.0 <= aspect_ratio <= 1.4:  # Square 2-line plates
            aspect_score = 1.3  # Good bonus for square 2-line plates
        elif 2.2 <= aspect_ratio <= 3.0:  # Between motorcycle and car
            aspect_score = 1.1
        elif 0.7 <= aspect_ratio <= 1.0:  # Very square 2-line plates
            aspect_score = 1.2  # Increased for very square motorcycle plates
        elif 2.0 <= aspect_ratio <= 5.0:  # Acceptable range
            aspect_score = 1.0
        else:
            aspect_score = 0.8
        
        # Method-specific bonuses with EXTREME MOTORCYCLE PRIORITY
        method_score = 1.0
        
        # EXTREME CASES - HIGHEST PRIORITY
        if method.startswith("ultra_tiny_"):  # ULTRA HIGH: Ultra-tiny distant plate detection
            method_score = 15.0
        elif method.startswith("indoor_gamma_"):  # ULTRA HIGH: Indoor lighting detection
            method_score = 12.0
        elif method.startswith("special_2line_"):  # ULTRA HIGH: Special 2-line with rider masking
            method_score = 11.0
        elif method.startswith("occluded_"):  # VERY HIGH: Occluded plate detection
            method_score = 10.0
        elif method.startswith("dark_"):  # VERY HIGH: Dark lighting detection
            method_score = 9.5
        
        # STANDARD MOTORCYCLE DETECTION
        elif method == "micro_motorcycle":  # HIGH: Micro motorcycle detection for distant plates
            method_score = 8.0
        elif method == "distant_motorcycle":  # HIGH: Distant motorcycle detection
            method_score = 7.0
        elif method.startswith("multiscale_"):  # HIGH: Multi-scale detection methods
            method_score = 6.0
        elif method == "motorcycle":  # HIGH: Standard motorcycle-specific detection
            method_score = 5.0
        elif method == "2line":  # HIGH: Two-line plate detection  
            method_score = 4.5
        
        # OTHER METHODS
        elif method == "bus":  # MEDIUM-HIGH: Bus-specific detection
            method_score = 3.0
        elif method == "bright":  # MEDIUM: Light text detection for buses
            method_score = 2.5
        elif method in ["dark", "contrast"]:  # MEDIUM: Good for regular license plates
            method_score = 2.0
        elif method == "adaptive":  # MEDIUM-LOW: Adaptive thresholding
            method_score = 1.8
        elif method == "edge":  # MEDIUM-LOW: Edge detection
            method_score = 1.6
        
        # Additional scoring bonus for license plate-like characteristics
        characteristic_bonus = 1.0
        
        # Bonus for typical license plate positioning (lower part of image, not edges)
        center_x = x + w/2
        center_y = y + h/2
        
        # Smart position-based characteristic bonus (motorcycle-aware)
        if (x <= 5 or y <= 5 or 
            (x + w) >= (img_width - 5) or (y + h) >= (img_height - 5)):
            characteristic_bonus = 0.01  # Extreme penalty for true edge detections
        
        # Enhanced bonus system for different plate types
        elif is_small_plate:  # Different rules for small motorcycle plates
            # Motorcycles can be anywhere in lower 2/3 of image
            if (img_height * 0.2 <= center_y <= img_height * 0.95 and  # Lower 75% of image
                img_width * 0.05 <= center_x <= img_width * 0.95):     # Almost entire width (motorcycles can be at sides)
                characteristic_bonus = 2.0  # Higher bonus for small plates in good positions
            else:  # Rules for larger car plates
                # Cars typically in lower 60% and more centered
                if (img_height * 0.3 <= center_y <= img_height * 0.9 and  # Lower 60% of image
                    img_width * 0.1 <= center_x <= img_width * 0.9):       # Central 80% horizontally
                    characteristic_bonus = 1.5
        
        # Enhanced size-based bonus for image coverage
        image_coverage = area / (img_width * img_height)
        if is_small_plate:
            # Motorcycle plates: 0.01% to 1% of image area
            if 0.0001 <= image_coverage <= 0.01:  
                characteristic_bonus *= 1.8  # Higher bonus for small plate coverage
            elif 0.01 <= image_coverage <= 0.02:  # Slightly larger motorcycles
                characteristic_bonus *= 1.5
        else:
            # Car plates: 0.2% to 3% of image area  
            if 0.002 <= image_coverage <= 0.03:
                characteristic_bonus *= 1.3
        
        return area_score * edge_penalty * position_score * size_score * aspect_score * method_score * characteristic_bonus
    
    unique_candidates.sort(key=license_plate_score, reverse=True)
    plate_candidates = unique_candidates[:10]
    
    # Draw results on original image
    result_img = img_np.copy()
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    
    for i, (x, y, w, h, area) in enumerate(plate_candidates):
        color = colors[i % len(colors)]
        cv2.rectangle(result_img, (x, y), (x+w, y+h), color, 3)
        cv2.putText(result_img, f'Candidate {i+1}', (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    phases['detection_result'] = result_img
    
    return phases, plate_candidates
