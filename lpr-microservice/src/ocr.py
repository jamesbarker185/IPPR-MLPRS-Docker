import cv2
import numpy as np
import re
import logging
from typing import List, Tuple, Dict, Optional

from .utils import enhance_plate_region

logger = logging.getLogger(__name__)

def load_ocr_model():
    """Load OCR model with fallback options"""
    # Try PaddleOCR first
    try:
        from paddleocr import PaddleOCR
        
        # Check PaddleOCR version and initialize accordingly
        ocr_model = PaddleOCR(
            use_angle_cls=True
        )
        
        logger.info("PaddleOCR loaded successfully!")
        return ocr_model, "paddleocr"
    except Exception as e:
        logger.warning(f"PaddleOCR failed to load: {str(e)}")
        return None, None

def validate_malaysian_plate_format(text: str) -> Tuple[bool, float, str]:
    """Validate if text matches Malaysian license plate formats"""
    if not text or len(text) < 3:
        return False, 0.0, "too_short"
    
    clean_text = re.sub(r'[^A-Z0-9]', '', text.upper())
    
    # Malaysian plate patterns - CORRECTED based on actual rules
    patterns = [
        # Standard format with suffix: A1A, ABC1234A, WCC9831A (start with alphabet(s), 1-4 integers, optional 1 alphabet)
        (r'^[A-Z]+[0-9]{1,4}[A-Z]$', 0.95, "standard_with_suffix"),
        # Standard format: A1, ABC1234, WCC9831, BLR83 (start with alphabet(s), 1-4 integers)
        (r'^[A-Z]+[0-9]{1,4}$', 0.90, "standard"),
    ]
    
    for pattern, confidence, format_type in patterns:
        if re.match(pattern, clean_text):
            # Additional validation for common Malaysian prefixes
            state_codes = ['A', 'B', 'C', 'D', 'F', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Z']
            
            if format_type in ["standard", "two_line"] and clean_text[0] in state_codes:
                confidence += 0.1  # Bonus for valid state code
            
            return True, confidence, format_type
    
    return False, 0.0, "invalid"

def detect_text_lines(roi_gray):
    """Detect horizontal text lines in the ROI for 2-line processing"""
    if roi_gray.size == 0:
        return []
    
    try:
        # Create horizontal projection
        horizontal_projection = np.sum(roi_gray < 128, axis=1)  # Sum dark pixels horizontally
        
        # Smooth the projection
        try:
            from scipy.ndimage import gaussian_filter1d
            smoothed = gaussian_filter1d(horizontal_projection, sigma=1.0)
        except ImportError:
            # Simple smoothing fallback
            kernel = np.array([0.25, 0.5, 0.25])
            smoothed = np.convolve(horizontal_projection, kernel, mode='same')
        
        # Find peaks (text lines)
        try:
            from scipy.signal import find_peaks
            peaks, properties = find_peaks(smoothed, height=np.max(smoothed) * 0.3, distance=len(smoothed)//4)
        except ImportError:
            # Fallback peak detection
            threshold = np.max(smoothed) * 0.3
            min_distance = len(smoothed) // 4
            peaks = []
            
            for i in range(min_distance, len(smoothed) - min_distance):
                if smoothed[i] > threshold:
                    # Check if it's a local maximum
                    is_peak = True
                    for j in range(max(0, i - min_distance), min(len(smoothed), i + min_distance)):
                        if smoothed[j] > smoothed[i]:
                            is_peak = False
                            break
                    if is_peak:
                        peaks.append(i)
            peaks = np.array(peaks)
        
        # Extract line regions
        lines = []
        for peak in peaks:
            # Find line boundaries
            start = peak
            end = peak
            
            # Expand backwards
            while start > 0 and smoothed[start] > np.max(smoothed) * 0.1:
                start -= 1
            
            # Expand forwards  
            while end < len(smoothed) - 1 and smoothed[end] > np.max(smoothed) * 0.1:
                end += 1
            
            if end - start > 5:  # Minimum line height
                lines.append((start, end))
        
        return lines
        
    except Exception:
        return []

def extract_license_plate_text_correct(ocr_model, image: np.ndarray) -> Optional[str]:

    if ocr_model is None:
        return None
        
    try:
        # Ensure image is in correct format
        if len(image.shape) == 3:
            # Convert RGB to BGR for PaddleOCR
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Get OCR results
        results = ocr_model.ocr(image)
        
        # Handle PaddleOCR results format
        if isinstance(results, list) and len(results) > 0:
            # Standard PaddleOCR format: list of list of [bbox, (text, confidence)]
            if results[0] is not None:
                texts_and_scores = []
                for i, detection in enumerate(results[0], 1):
                    if len(detection) >= 2:
                        # detection[1] should be (text, confidence)
                        ocr_result = detection[1]
                        if isinstance(ocr_result, (tuple, list)) and len(ocr_result) >= 2:
                            text, confidence = ocr_result[0], ocr_result[1]
                            texts_and_scores.append((text, confidence))
                
                if texts_and_scores:
                    # Sort by confidence to get reliable texts first
                    texts_and_scores.sort(key=lambda x: x[1], reverse=True)
                    
                    # Try to find single complete plate text first
                    for text, score in texts_and_scores:
                        clean_text = re.sub(r'[^A-Z0-9]', '', text.strip().upper())
                        if len(clean_text) >= 4 and score > 0.6:  # LOWERED: More lenient for single plates
                            # VALIDATE FORMAT before accepting single plate
                            is_valid, format_conf, format_type = validate_malaysian_plate_format(clean_text)
                            
                            if is_valid:
                                return clean_text
                    
                    # ENHANCED 2-LINE PROCESSING: Better line detection and combination
                    if len(texts_and_scores) >= 2:
                        # First, try to detect text lines using image analysis
                        if len(image.shape) == 3:
                            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                        else:
                            gray = image
                        
                        text_lines = detect_text_lines(gray)
                        
                        # Get text detections along with their bounding boxes
                        bbox_texts = []
                        for i, detection in enumerate(results[0][:len(texts_and_scores)]):
                            if len(detection) >= 2:
                                bbox = detection[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                                ocr_result = detection[1]
                                if isinstance(ocr_result, (tuple, list)) and len(ocr_result) >= 2:
                                    text, confidence = ocr_result[0], ocr_result[1]
                                    clean_text = re.sub(r'[^A-Z0-9]', '', text.strip().upper())
                                    
                                    if len(clean_text) >= 1 and confidence > 0.2:  # LOWERED: Even more lenient for 2-line parts
                                        # Calculate center coordinates with safety checks
                                        try:
                                            if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                                                # bbox should be [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                                                center_y = sum([point[1] for point in bbox if isinstance(point, (list, tuple)) and len(point) >= 2]) / 4
                                                center_x = sum([point[0] for point in bbox if isinstance(point, (list, tuple)) and len(point) >= 2]) / 4
                                                
                                                # Calculate text area for weighting
                                                x_coords = [point[0] for point in bbox if isinstance(point, (list, tuple)) and len(point) >= 2]
                                                y_coords = [point[1] for point in bbox if isinstance(point, (list, tuple)) and len(point) >= 2]
                                                if x_coords and y_coords:
                                                    width = max(x_coords) - min(x_coords)
                                                    height = max(y_coords) - min(y_coords)
                                                    area = width * height
                                                    
                                                    bbox_texts.append((clean_text, confidence, center_y, center_x, area))
                                        except (IndexError, TypeError, ValueError, ZeroDivisionError):
                                            # Skip this bbox if coordinates are malformed
                                            pass
                        
                        if len(bbox_texts) >= 2:
                            # Sort by vertical position (top to bottom)
                            bbox_texts.sort(key=lambda x: x[2])
                            
                            # Try different combination strategies
                            best_combination = None
                            best_confidence = 0
                            
                            # Strategy 1: Simple top-bottom combination
                            for i in range(len(bbox_texts)-1):
                                for j in range(i+1, min(len(bbox_texts), i+3)):  # Try next 2 candidates
                                    text1, conf1, y1, x1, area1 = bbox_texts[i]
                                    text2, conf2, y2, x2, area2 = bbox_texts[j]
                                    
                                    # Check if texts are vertically aligned (2-line format)
                                    vertical_distance = abs(y2 - y1)
                                    horizontal_overlap = min(abs(x1 - x2), gray.shape[1] * 0.3)
                                    
                                    if vertical_distance > 5 and horizontal_overlap < gray.shape[1] * 0.5:
                                        # Determine order based on position and typical Malaysian formats
                                        if y1 < y2:  # text1 is above text2
                                            combined = text1 + text2
                                        else:  # text2 is above text1
                                            combined = text2 + text1
                                        
                                        # Validate combination
                                        is_valid, format_conf, _ = validate_malaysian_plate_format(combined)
                                        
                                        if is_valid:
                                            combo_confidence = (conf1 + conf2) / 2 * format_conf
                                            if combo_confidence > best_confidence:
                                                best_combination = combined
                                                best_confidence = combo_confidence
                            
                            # Strategy 2: Fallback - simple concatenation of top 2
                            if not best_combination and len(bbox_texts) >= 2:
                                combined_text = bbox_texts[0][0] + bbox_texts[1][0]
                                avg_confidence = (bbox_texts[0][1] + bbox_texts[1][1]) / 2
                                
                                if len(combined_text) >= 3 and avg_confidence > 0.3:  # LOWERED: More lenient fallback
                                    best_combination = combined_text
                                    best_confidence = avg_confidence
                            
                            if best_combination and best_confidence > 0.3:  # LOWERED: Accept more combinations
                                return best_combination
                    
                    # Fallback: use best single detection
                    best_text = None
                    highest_score = 0
                    
                    for text, score in texts_and_scores:
                        clean_text = re.sub(r'[^A-Z0-9]', '', text.strip().upper())
                        if len(clean_text) >= 2 and score > highest_score:
                            best_text = clean_text
                            highest_score = score
                    
                    if best_text and len(best_text) >= 3:
                        return best_text
        
        return None
            
    except Exception as e:
        logger.error(f"OCR Error: {str(e)}")
        return None

def extract_plate_text(image: np.ndarray, ocr_model, ocr_engine: str) -> str:
    """Extract text from license plate using available OCR engine"""
    if ocr_engine == "paddleocr" and ocr_model:
        result = extract_license_plate_text_correct(ocr_model, image)
        return result if result else ""
    else:
        return ""

def ocr_verification_pipeline(ocr_model, ocr_engine: str, candidates: List, phases_dict: Dict) -> List[Tuple]:
    """Run OCR verification on top candidates using multiple phases for better results"""
    if not candidates or ocr_model is None:
        return [(c[0], c[1], c[2], c[3], c[4], "", 0.0) for c in candidates]
    
    verified_candidates = []
    
    # Define OCR phases to try (in order of preference)
    ocr_phases = [
        ("restored", "Phase 3: Bilateral filtered"),
        ("enhanced", "Phase 2: Histogram equalized"), 
        ("color_processed", "Phase 4: HSV Value channel")
    ]
    
    # Process top candidates (expanded to match UI display)
    for i, (x, y, w, h, area) in enumerate(candidates[:10]):
        try:
            best_text = ""
            best_confidence = 0.0
            best_phase = ""
            all_attempts = []
            
            # Try OCR on multiple phases
            for phase_key, phase_name in ocr_phases:
                if phase_key not in phases_dict:
                    continue
                    
                try:
                    # Extract ROI from this phase
                    phase_image = phases_dict[phase_key]
                    enhanced_roi = enhance_plate_region(phase_image, x, y, w, h)
                    
                    # Run OCR on this phase
                    extracted_text = extract_plate_text(enhanced_roi, ocr_model, ocr_engine)
                    
                    if extracted_text:
                        # Validate format
                        is_valid, format_confidence, format_type = validate_malaysian_plate_format(extracted_text)
                        
                        # Calculate confidence for this attempt
                        text_length_score = min(1.0, len(extracted_text) / 6.0)
                        attempt_confidence = text_length_score * 0.5 + format_confidence * 0.5
                        
                        # Bonus for valid Malaysian plate format
                        if is_valid:
                            attempt_confidence *= 1.5
                        
                        all_attempts.append({
                            'text': extracted_text,
                            'confidence': attempt_confidence,
                            'phase': phase_name,
                            'is_valid': is_valid,
                            'format_type': format_type
                        })
                        
                        
                        # Update best result if this is better
                        if attempt_confidence > best_confidence:
                            best_text = extracted_text
                            best_confidence = attempt_confidence
                            best_phase = phase_name
                            
                except Exception as phase_error:
                    continue
            
            # MAJORITY VOTING: When scores are similar, use majority vote
            if all_attempts:
                # Group attempts by confidence range (within 0.1 of each other)
                confidence_groups = {}
                for attempt in all_attempts:
                    conf_key = round(attempt['confidence'], 1)  # Round to nearest 0.1
                    if conf_key not in confidence_groups:
                        confidence_groups[conf_key] = []
                    confidence_groups[conf_key].append(attempt)
                
                # Find the highest confidence group
                max_conf_key = max(confidence_groups.keys()) if confidence_groups else 0
                top_group = confidence_groups.get(max_conf_key, [])
                
                # Apply majority voting within the top confidence group
                if len(top_group) > 1:
                    # Count occurrences of each text result
                    text_votes = {}
                    for attempt in top_group:
                        text = attempt['text']
                        if text not in text_votes:
                            text_votes[text] = {'count': 0, 'total_conf': 0, 'attempts': []}
                        text_votes[text]['count'] += 1
                        text_votes[text]['total_conf'] += attempt['confidence']
                        text_votes[text]['attempts'].append(attempt)
                    
                    # Find majority winner
                    max_votes = max(text_votes[text]['count'] for text in text_votes)
                    majority_candidates = [text for text in text_votes if text_votes[text]['count'] == max_votes]
                    
                    if len(majority_candidates) == 1:
                        # Clear majority winner
                        majority_text = majority_candidates[0]
                        majority_data = text_votes[majority_text]
                        best_text = majority_text
                        best_confidence = majority_data['total_conf'] / majority_data['count']
                        best_phase = majority_data['attempts'][0]['phase']
                    
                    elif len(majority_candidates) > 1:
                        # Tie-breaker: prefer longer text (more complete detection)
                        longest_text = max(majority_candidates, key=len)
                        majority_data = text_votes[longest_text]
                        best_text = longest_text
                        best_confidence = majority_data['total_conf'] / majority_data['count']
                        best_phase = majority_data['attempts'][0]['phase']
                
            else:
                pass
            
            # Calculate final OCR confidence
            ocr_confidence = best_confidence
            
            # Penalty for obviously wrong text
            if best_text and (len(best_text) > 10 or any(char in best_text for char in ['@', '#', '$', '%'])):
                ocr_confidence *= 0.1
            
            # Boost area score based on OCR success
            boosted_area = area
            is_valid = any(a['is_valid'] for a in all_attempts)
            
            if is_valid and best_text:
                # MASSIVE boost to ensure valid plates become Candidate 1
                boosted_area *= (1.0 + ocr_confidence * 50.0)  # INCREASED from 3.0 to 50.0
            elif best_text and len(best_text) >= 3:
                boosted_area *= (1.0 + ocr_confidence * 5.0)  # INCREASED from 1.0 to 5.0
            else:
                boosted_area *= 0.1  # INCREASED penalty to push non-readable candidates down
            
            verified_candidates.append((x, y, w, h, boosted_area, best_text, ocr_confidence))
            
        except Exception as e:
            verified_candidates.append((x, y, w, h, area * 0.3, "", 0.0))  # Penalty for OCR failure
    
    # GUARANTEE: Valid license plates always come first
    valid_plates = []
    other_candidates = []
    
    for candidate in verified_candidates:
        x, y, w, h, boosted_area, text, confidence = candidate
        
        # Check if this candidate has a valid Malaysian plate text
        if text:
            is_valid, _, _ = validate_malaysian_plate_format(text)
            if is_valid:
                valid_plates.append(candidate)
            else:
                other_candidates.append(candidate)
        else:
            other_candidates.append(candidate)
    
    # Sort each group by boosted area score
    valid_plates.sort(key=lambda x: x[4], reverse=True)
    other_candidates.sort(key=lambda x: x[4], reverse=True)
    
    # Combine: valid plates first, then others
    final_candidates = valid_plates + other_candidates
    
    return final_candidates
