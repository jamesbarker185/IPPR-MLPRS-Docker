import cv2
import numpy as np
from typing import List, Tuple

def analyze_color_characteristics(roi_color):
    """Analyze color characteristics specific to Malaysian license plates"""
    if roi_color.size == 0 or len(roi_color.shape) != 3:
        return False, 0.0
    
    try:
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(roi_color, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        
        # Check for dominant light background (most common)
        light_pixels = np.sum(v > 180)  # Bright pixels
        dark_pixels = np.sum(v < 75)    # Dark pixels
        total_pixels = roi_color.shape[0] * roi_color.shape[1]
        
        light_ratio = light_pixels / total_pixels
        dark_ratio = dark_pixels / total_pixels
        
        # Pattern 1: Light background with dark text (70%+ light background)
        is_light_plate = light_ratio > 0.6 and dark_ratio > 0.05
        
        # Pattern 2: Dark background with light text (60%+ dark background)
        is_dark_plate = dark_ratio > 0.5 and light_ratio > 0.05
        
        # Pattern 3: Moderate contrast (good mix of light and dark)
        is_contrast_plate = 0.2 <= light_ratio <= 0.8 and 0.1 <= dark_ratio <= 0.6
        
        # Calculate color uniformity (plates have relatively uniform backgrounds)
        bg_pixels = roi_color[v > np.median(v)]  # Background pixels
        if len(bg_pixels) > 10:
            bg_std = np.std(bg_pixels)
            color_uniformity = 1.0 / (1.0 + bg_std / 50.0)  # Lower std = higher uniformity
        else:
            color_uniformity = 0.0
        
        is_plate_color = is_light_plate or is_dark_plate or is_contrast_plate
        confidence = color_uniformity * (0.8 if is_light_plate else 0.6 if is_dark_plate else 0.4)
        
        return is_plate_color, confidence
        
    except Exception:
        return False, 0.0

def analyze_texture_characteristics(roi_gray):
    """Analyze texture patterns typical of license plates"""
    if roi_gray.size == 0:
        return False, 0.0
    
    try:
        # License plates have relatively uniform texture with text patterns
        
        # 1. Calculate gradient magnitude for texture analysis
        grad_x = cv2.Sobel(roi_gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(roi_gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # 2. Check for regular text patterns
        # License plates have regular character spacing
        horizontal_profile = np.mean(gradient_magnitude, axis=0)
        
        # Find peaks in horizontal profile (character boundaries)
        try:
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(horizontal_profile, height=np.mean(horizontal_profile) * 0.5)
        except ImportError:
            # Fallback peak detection without scipy
            threshold = np.mean(horizontal_profile) * 0.5
            peaks = []
            for i in range(1, len(horizontal_profile) - 1):
                if (horizontal_profile[i] > threshold and 
                    horizontal_profile[i] > horizontal_profile[i-1] and 
                    horizontal_profile[i] > horizontal_profile[i+1]):
                    peaks.append(i)
            peaks = np.array(peaks)
        
        # Regular spacing indicates text characters
        if len(peaks) >= 2:
            spacings = np.diff(peaks)
            spacing_regularity = 1.0 - (np.std(spacings) / np.mean(spacings)) if np.mean(spacings) > 0 else 0.0
            spacing_regularity = max(0.0, min(1.0, spacing_regularity))
        else:
            spacing_regularity = 0.0
        
        # 3. Texture uniformity (backgrounds should be relatively uniform)
        texture_variance = np.var(roi_gray)
        texture_score = 1.0 / (1.0 + abs(texture_variance - 1500) / 1000.0)  # Optimal around 1500
        
        is_plate_texture = spacing_regularity > 0.3 or texture_score > 0.5
        confidence = (spacing_regularity * 0.6 + texture_score * 0.4)
        
        return is_plate_texture, confidence
        
    except Exception:
        return False, 0.0

def match_plate_template(roi_gray):
    """Template matching for Malaysian license plate characteristics"""
    if roi_gray.size == 0:
        return False, 0.0
    
    try:
        # Create simple templates for character patterns
        h, w = roi_gray.shape
        
        # Template 1: Single line text pattern
        template_1line = np.ones((max(20, h//3), max(60, w//2)), dtype=np.uint8) * 255
        template_1line[h//6:h//6+h//6, :] = 0  # Dark text stripe
        
        # Template 2: Two line text pattern  
        template_2line = np.ones((max(30, h//2), max(40, w//3)), dtype=np.uint8) * 255
        template_2line[h//8:h//8+h//8, :] = 0      # Top text line
        template_2line[h//2:h//2+h//8, :] = 0      # Bottom text line
        
        # Resize templates to match ROI size
        template_1line = cv2.resize(template_1line, (min(w, 100), min(h, 40)))
        template_2line = cv2.resize(template_2line, (min(w, 80), min(h, 60)))
        
        # Resize ROI for template matching
        roi_resized = cv2.resize(roi_gray, (template_1line.shape[1], template_1line.shape[0]))
        
        # Match templates
        match_1line = cv2.matchTemplate(roi_resized, template_1line, cv2.TM_CCOEFF_NORMED)
        match_2line = cv2.matchTemplate(roi_resized, template_2line, cv2.TM_CCOEFF_NORMED)
        
        confidence_1line = np.max(match_1line) if match_1line.size > 0 else 0.0
        confidence_2line = np.max(match_2line) if match_2line.size > 0 else 0.0
        
        best_confidence = max(confidence_1line, confidence_2line)
        is_template_match = best_confidence > 0.3
        
        return is_template_match, best_confidence
        
    except Exception:
        return False, 0.0

def find_plate_candidates_from_binary(binary_image: np.ndarray, original_image: np.ndarray = None) -> List[Tuple[int, int, int, int, float]]:
    """Enhanced helper function to find candidates from binary image with smart filtering"""
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return []
    
    candidates = []
    img_area = binary_image.shape[0] * binary_image.shape[1]
    
    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        
        if h == 0 or w == 0:
            continue
            
        # Improved filtering approach with percentage-based sizing
        aspect_ratio = w / float(h)
        area_ratio = (w * h) / img_area
        
        # 1. FIXED: Much more lenient aspect ratio check (includes ALL valid plates)
        valid_single_line = 1.8 <= aspect_ratio <= 6.0    # Standard car plates (EXPANDED)
        valid_two_line = 0.7 <= aspect_ratio <= 3.0       # 2-line plates (EXPANDED)
        valid_square_2line = 0.5 <= aspect_ratio <= 1.5   # Square plates (EXPANDED)
        valid_motorcycle = 1.0 <= aspect_ratio <= 2.8     # Motorcycles (EXPANDED)
        valid_wide_plates = 2.8 <= aspect_ratio <= 4.0    # Wide plates (EXPANDED)
        valid_very_wide = 4.0 <= aspect_ratio <= 7.0      # Very wide plates (NEW)
        valid_aspect = valid_single_line or valid_two_line or valid_square_2line or valid_motorcycle or valid_wide_plates or valid_very_wide
        
        # 2. FIXED: Balanced size check (works for both cars and motorcycles)  
        valid_size = 0.00001 <= area_ratio <= 0.20  # VERY LENIENT for all plate types
        
        # 3. FIXED: More reasonable minimum dimensions (not too restrictive)
        valid_dimensions = w > 12 and h > 4  # REDUCED further to catch more plates
        
        # 4. FIXED: More lenient maximum dimensions 
        max_w = binary_image.shape[1] * 0.8  # Max 80% of image width (INCREASED)
        max_h = binary_image.shape[0] * 0.6  # Max 60% of image height (INCREASED)
        valid_max_dimensions = w <= max_w and h <= max_h
        
        if not (valid_aspect and valid_size and valid_dimensions and valid_max_dimensions):
            continue
            
        # 5. Rectangularity and shape quality checks
        extent = area / (w * h) if (w * h) > 0 else 0
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        # Check if contour is approximately rectangular
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        # Approximate contour to check for rectangular shape
        epsilon = 0.02 * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Filter for license plate shape characteristics (FIXED: More lenient)
        is_rectangular = len(approx) >= 3 and len(approx) <= 12  # VERY flexible for all plate types
        good_circularity = 0.02 <= circularity <= 0.98  # VERY RELAXED for all detections
        good_extent = extent > 0.20  # VERY RELAXED for all plate types
        good_solidity = solidity > 0.4   # VERY RELAXED for all conditions
        
        if not (good_extent and good_solidity and is_rectangular and good_circularity):
            continue
            
        # 6. SMART FILTERING: Progressive multi-stage validation
        confidence_score = 0.0
        
        if original_image is not None:
            # Extract regions for analysis
            roi_gray = binary_image[y:y+h, x:x+w] if len(binary_image.shape) == 2 else cv2.cvtColor(binary_image[y:y+h, x:x+w], cv2.COLOR_RGB2GRAY)
            roi_color = original_image[y:y+h, x:x+w] if len(original_image.shape) == 3 else cv2.cvtColor(original_image[y:y+h, x:x+w], cv2.COLOR_GRAY2RGB)
            
            # Stage 1: Color Analysis
            is_color_match, color_confidence = analyze_color_characteristics(roi_color)
            confidence_score += color_confidence * 0.3
            
            # Stage 2: Texture Analysis  
            is_texture_match, texture_confidence = analyze_texture_characteristics(roi_gray)
            confidence_score += texture_confidence * 0.3
            
            # Stage 3: Template Matching
            is_template_match, template_confidence = match_plate_template(roi_gray)
            confidence_score += template_confidence * 0.4
            
            # Progressive filtering: Only reject if ALL methods fail
            total_matches = sum([is_color_match, is_texture_match, is_template_match])
            
            # Reject only if confidence is very low AND no method matches
            if confidence_score < 0.1 and total_matches == 0:
                continue  # Reject obvious non-plates
            
            # Boost area score based on confidence
            area = area * (1.0 + confidence_score)  # Confidence boost
        
        candidates.append((x, y, w, h, area))
    
    return candidates

def detect_license_plate_regions(image: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
    """
    Detect license plate regions using multiple preprocessing approaches
    
    Args:
        image: Input grayscale image (NOT binary)
        
    Returns:
        List of tuples (x, y, w, h, area) for detected plate candidates
    """
    all_candidates = []
    
    # Method 1: Adaptive thresholding
    try:
        adaptive_thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        candidates1 = find_plate_candidates_from_binary(adaptive_thresh, image)
        all_candidates.extend([(c[0], c[1], c[2], c[3], c[4], "adaptive") for c in candidates1])
    except:
        pass
    
    # Method 2: Dark region detection (license plates are typically dark)
    try:
        # Invert image to find dark regions as white
        inverted = cv2.bitwise_not(image)
        _, dark_thresh = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        candidates2 = find_plate_candidates_from_binary(dark_thresh, image)
        all_candidates.extend([(c[0], c[1], c[2], c[3], c[4], "dark") for c in candidates2])
    except:
        pass
    
    # Method 2b: Standard Otsu thresholding
    try:
        _, otsu_thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        candidates2b = find_plate_candidates_from_binary(otsu_thresh, image)
        all_candidates.extend([(c[0], c[1], c[2], c[3], c[4], "otsu") for c in candidates2b])
    except:
        pass
    
    # Method 3: Enhanced edge detection for license plates
    try:
        # Use bilateral filter before edge detection to reduce noise but keep edges
        bilateral = cv2.bilateralFilter(image, 11, 17, 17)
        edges = cv2.Canny(bilateral, 30, 100)
        
        # Use wider horizontal kernel to connect license plate characters
        kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (12, 3))  # Wider to connect "AAT" and "40"
        morph = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_rect)
        
        # Additional horizontal dilation to connect characters
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 2))  # Wider horizontal connection
        morph = cv2.dilate(morph, kernel_dilate, iterations=2)
        
        candidates3 = find_plate_candidates_from_binary(morph, image)
        all_candidates.extend([(c[0], c[1], c[2], c[3], c[4], "edge") for c in candidates3])
    except:
        pass
    
    # Method 4: License plate specific detection (dark regions with high contrast)
    try:
        # Look for regions that are darker than average but have high local contrast
        mean_intensity = np.mean(image)
        
        # Create mask for dark regions
        dark_mask = image < (mean_intensity * 0.7)
        dark_mask = dark_mask.astype(np.uint8) * 255
        
        # Apply morphological operations to connect license plate characters
        # Use wider horizontal kernel to connect characters on the same plate
        kernel_connect = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))  # Wider to connect characters
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel_connect)
        
        # Clean up noise with smaller kernel
        kernel_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel_clean)
        
        candidates4 = find_plate_candidates_from_binary(dark_mask, image)
        all_candidates.extend([(c[0], c[1], c[2], c[3], c[4], "contrast") for c in candidates4])
    except:
        pass
    
    # Method 5: Light text on dark background (bus plates)
    try:
        # Look for bright regions (light text) on darker backgrounds
        mean_intensity = np.mean(image)
        
        # Create mask for bright regions (light text) - more aggressive threshold
        bright_mask = image > (mean_intensity * 1.1)  # Lower threshold to catch more text
        bright_mask = bright_mask.astype(np.uint8) * 255
        
        # Apply morphological operations to connect light text characters
        kernel_connect = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 6))  # Even wider for bus plates
        bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, kernel_connect)
        
        # Additional dilation to ensure text connection
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
        bright_mask = cv2.dilate(bright_mask, kernel_dilate, iterations=1)
        
        # Clean up with opening
        kernel_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 5))
        bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_OPEN, kernel_clean)
        
        candidates5 = find_plate_candidates_from_binary(bright_mask, image)
        all_candidates.extend([(c[0], c[1], c[2], c[3], c[4], "bright") for c in candidates5])
    except:
        pass
    
    # Method 6: Bus-specific license plate detection
    try:
        height, width = image.shape[:2]
        
        # Focus on lower portion of image where bus plates typically are
        lower_third = image[height//3:, :]  # Bottom 2/3 of image
        
        # Use very aggressive thresholding for bus plates
        _, bus_thresh = cv2.threshold(lower_third, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Very wide morphological operations for bus text
        kernel_wide = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 8))  # Extra wide for bus plates
        bus_processed = cv2.morphologyEx(bus_thresh, cv2.MORPH_CLOSE, kernel_wide)
        
        # Additional connection with dilation
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 5))
        bus_processed = cv2.dilate(bus_processed, kernel_dilate, iterations=2)
        
        # Clean up small noise
        kernel_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 6))
        bus_processed = cv2.morphologyEx(bus_processed, cv2.MORPH_OPEN, kernel_clean)
        
        # Find candidates in the processed lower portion
        bus_candidates = find_plate_candidates_from_binary(bus_processed, lower_third)
        
        # Adjust y-coordinates back to full image coordinates
        adjusted_bus_candidates = []
        for x, y, w, h, area in bus_candidates:
            adjusted_y = y + height//3  # Add offset for lower third
            adjusted_bus_candidates.append((x, adjusted_y, w, h, area, "bus"))
        
        all_candidates.extend(adjusted_bus_candidates)
    except:
        pass
    
    # Method 7: Enhanced two-line license plate detection (motorcycles)
    try:
        # Apply adaptive threshold optimized for 2-line plates
        adaptive_2line = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 3)
        
        # MOTORCYCLE-SPECIFIC: Use vertical morphological operations to connect stacked text
        kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 8))  # Tall kernel for vertical connection
        morph_2line = cv2.morphologyEx(adaptive_2line, cv2.MORPH_CLOSE, kernel_vertical)
        
        # Also apply horizontal connection within each line
        kernel_horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 3))  # Wide kernel for horizontal connection
        morph_2line = cv2.morphologyEx(morph_2line, cv2.MORPH_CLOSE, kernel_horizontal)
        
        # Additional morphological operations to merge the two lines
        kernel_merge = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 12))  # Medium width, taller for 2-line merging
        morph_2line = cv2.morphologyEx(morph_2line, cv2.MORPH_CLOSE, kernel_merge)
        
        # Clean up noise while preserving main structures
        kernel_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
        morph_2line = cv2.morphologyEx(morph_2line, cv2.MORPH_OPEN, kernel_clean)
        
        candidates_2line = find_plate_candidates_from_binary(morph_2line, image)
        all_candidates.extend([(c[0], c[1], c[2], c[3], c[4], "2line") for c in candidates_2line])
    except:
        pass
    
    # Method 8: ENHANCED MOTORCYCLE-SPECIFIC detection (small, square plates)
    try:
        # Use smaller, more aggressive thresholding for small motorcycle plates
        motorcycle_thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 3)
        
        # STAGE 1: Micro-motorcycles (very small distant plates)
        # Use tiny morphological kernels for very small text
        kernel_micro_h = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))  # Tiny horizontal connection
        kernel_micro_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 4))  # Tiny vertical connection
        
        micro_processed = cv2.morphologyEx(motorcycle_thresh, cv2.MORPH_CLOSE, kernel_micro_h)
        micro_processed = cv2.morphologyEx(micro_processed, cv2.MORPH_CLOSE, kernel_micro_v)
        
        # Very light final connection
        kernel_micro_final = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 5))
        micro_processed = cv2.morphologyEx(micro_processed, cv2.MORPH_CLOSE, kernel_micro_final)
        
        candidates_micro = find_plate_candidates_from_binary(micro_processed, image)
        all_candidates.extend([(c[0], c[1], c[2], c[3], c[4], "micro_motorcycle") for c in candidates_micro])
        
        # STAGE 2: Regular motorcycles (standard size)
        # Use slightly larger kernels for normal motorcycle plates
        kernel_moto_h = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 2))  # Small horizontal connection
        kernel_moto_v = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 5))  # Small vertical connection
        
        moto_processed = cv2.morphologyEx(motorcycle_thresh, cv2.MORPH_CLOSE, kernel_moto_h)
        moto_processed = cv2.morphologyEx(moto_processed, cv2.MORPH_CLOSE, kernel_moto_v)
        
        # Final connection with small kernel
        kernel_final = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 6))
        moto_processed = cv2.morphologyEx(moto_processed, cv2.MORPH_CLOSE, kernel_final)
        
        # Very light cleanup to preserve small plates
        kernel_cleanup = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        moto_processed = cv2.morphologyEx(moto_processed, cv2.MORPH_OPEN, kernel_cleanup)
        
        candidates_motorcycle = find_plate_candidates_from_binary(moto_processed, image)
        all_candidates.extend([(c[0], c[1], c[2], c[3], c[4], "motorcycle") for c in candidates_motorcycle])
        
    except:
        pass
    
    # Method 9: MULTI-SCALE MOTORCYCLE detection (handles various distances)
    try:
        # Create multiple scales to handle plates at different distances
        h_img, w_img = image.shape[:2]
        
        # Scale 1: Original size for close motorcycles
        scale1_thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 2)
        
        # Use progressive kernels for different text sizes
        for kernel_size in [(2, 1), (3, 2), (4, 3)]:  # Very small kernels
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
            processed = cv2.morphologyEx(scale1_thresh, cv2.MORPH_CLOSE, kernel)
            
            # Add vertical connection for 2-line plates
            kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_size[1] + 2))
            processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel_v)
            
            candidates_scale = find_plate_candidates_from_binary(processed, image)
            all_candidates.extend([(c[0], c[1], c[2], c[3], c[4], f"multiscale_{kernel_size[0]}x{kernel_size[1]}") for c in candidates_scale])
            
        # Scale 2: Handle very small distant plates with upscaling
        if min(h_img, w_img) > 200:  # Only if image is large enough
            # Create a focused region for distant plate detection (lower 2/3 of image)
            roi_y_start = h_img // 3
            roi = image[roi_y_start:, :]
            
            # Apply more aggressive thresholding for distant small plates
            distant_thresh = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 1)
            
            # Use the smallest possible kernels for distant plates
            kernel_tiny = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
            distant_processed = cv2.morphologyEx(distant_thresh, cv2.MORPH_CLOSE, kernel_tiny)
            
            # Add minimal vertical connection
            kernel_tiny_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
            distant_processed = cv2.morphologyEx(distant_processed, cv2.MORPH_CLOSE, kernel_tiny_v)
            
            # Find candidates in ROI and adjust coordinates
            roi_candidates = find_plate_candidates_from_binary(distant_processed, roi)
            for x, y, w, h, area in roi_candidates:
                adjusted_y = y + roi_y_start  # Adjust for ROI offset
                all_candidates.append((x, adjusted_y, w, h, area, "distant_motorcycle"))
                
    except:
        pass
    
    # Method 10: EXTREME CASES detection (for very challenging motorcycle plates)
    try:
        
        # CASE 1: Ultra-small distant plates (like KAA17)
        # Use minimal thresholding to catch faint distant plates
        _, ultra_thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Ultra-minimal morphological operations
        kernel_ultra = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))  # Minimal kernel
        ultra_processed = cv2.morphologyEx(ultra_thresh, cv2.MORPH_CLOSE, kernel_ultra)
        
        # Try to connect very small text pieces with tiny kernels
        for tiny_size in [(2, 1), (1, 2), (3, 1), (1, 3)]:
            tiny_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, tiny_size)
            temp_processed = cv2.morphologyEx(ultra_processed, cv2.MORPH_CLOSE, tiny_kernel)
            
            candidates_ultra = find_plate_candidates_from_binary(temp_processed, image)
            all_candidates.extend([(c[0], c[1], c[2], c[3], c[4], f"ultra_tiny_{tiny_size[0]}x{tiny_size[1]}") for c in candidates_ultra])
        
        # CASE 2: Dark/low-light plates (like GT41)
        # Enhanced histogram equalization for dark images
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4,4))
        enhanced_dark = clahe.apply(image)
        
        # Apply multiple thresholding approaches for dark images
        dark_methods = [
            (cv2.THRESH_BINARY, "dark_binary"),
            (cv2.THRESH_BINARY_INV, "dark_binary_inv"), 
            (cv2.ADAPTIVE_THRESH_GAUSSIAN_C, "dark_adaptive_gauss"),
            (cv2.ADAPTIVE_THRESH_MEAN_C, "dark_adaptive_mean")
        ]
        
        for thresh_type, method_name in dark_methods:
            try:
                if thresh_type in [cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV]:
                    _, dark_thresh = cv2.threshold(enhanced_dark, 0, 255, thresh_type + cv2.THRESH_OTSU)
                else:
                    dark_thresh = cv2.adaptiveThreshold(enhanced_dark, 255, thresh_type, cv2.THRESH_BINARY, 3, 1)
                
                # Very light morphology for dark images
                kernel_dark = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
                dark_processed = cv2.morphologyEx(dark_thresh, cv2.MORPH_CLOSE, kernel_dark)
                
                candidates_dark = find_plate_candidates_from_binary(dark_processed, image)
                all_candidates.extend([(c[0], c[1], c[2], c[3], c[4], method_name) for c in candidates_dark])
            except:
                continue
        
        # CASE 3: Heavily occluded plates (like MBQ73)
        # Use very permissive shape filtering
        # Apply gentle gaussian blur to connect fragmented text
        blurred = cv2.GaussianBlur(image, (3, 3), 1.0)
        _, occluded_thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Try different dilation strategies to connect broken text
        for dilation_size in [(1, 2), (2, 1), (2, 2), (3, 2), (2, 3)]:
            kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, dilation_size)
            occluded_processed = cv2.dilate(occluded_thresh, kernel_dilate, iterations=1)
            
            # Very light erosion to clean up
            kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
            occluded_processed = cv2.erode(occluded_processed, kernel_erode, iterations=1)
            
            candidates_occluded = find_plate_candidates_from_binary(occluded_processed, image)
            all_candidates.extend([(c[0], c[1], c[2], c[3], c[4], f"occluded_{dilation_size[0]}x{dilation_size[1]}") for c in candidates_occluded])
        
        # CASE 4: Two-line plates with rider interference (like BNW76)
        # Focus on vertical text connection with rider masking
        
        # Create a mask to reduce rider interference (focus on lower portion)
        # Use fallback if image is binary (no rider masking possible)
        if len(image.shape) == 2:
            masked_image = image.copy()
        else:
            h_img, w_img = image.shape[:2]
            mask = np.zeros_like(image)
            # Focus on bottom 60% where license plates typically are, avoiding rider torso
            mask[int(h_img * 0.4):, int(w_img * 0.1):int(w_img * 0.9)] = 255
            masked_image = cv2.bitwise_and(image, mask)
        
        # Apply aggressive adaptive thresholding on masked region
        masked_thresh = cv2.adaptiveThreshold(masked_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 7, 2)
        
        # Specialized 2-line connection with various vertical kernels
        for v_height in range(3, 12, 2):  # Try different vertical connection heights
            kernel_2line = cv2.getStructuringElement(cv2.MORPH_RECT, (2, v_height))
            two_line_processed = cv2.morphologyEx(masked_thresh, cv2.MORPH_CLOSE, kernel_2line)
            
            # Add horizontal connection within each line
            kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 1))
            two_line_processed = cv2.morphologyEx(two_line_processed, cv2.MORPH_CLOSE, kernel_h)
            
            candidates_2line_special = find_plate_candidates_from_binary(two_line_processed, image)
            all_candidates.extend([(c[0], c[1], c[2], c[3], c[4], f"special_2line_v{v_height}") for c in candidates_2line_special])
        
        # CASE 5: Indoor/artificial lighting plates (like WCQ7)
        # Enhanced gamma correction for indoor lighting
        for gamma in [0.5, 0.7, 1.3, 1.5]:  # Different gamma values
            gamma_corrected = np.array(255 * (image / 255) ** gamma, dtype='uint8')
            
            # Apply strong adaptive thresholding
            indoor_thresh = cv2.adaptiveThreshold(gamma_corrected, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 5, 1)
            
            # Minimal morphology to preserve small text
            kernel_indoor = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
            indoor_processed = cv2.morphologyEx(indoor_thresh, cv2.MORPH_CLOSE, kernel_indoor)
            
            # Add tiny vertical connection for 2-line indoor plates
            kernel_indoor_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
            indoor_processed = cv2.morphologyEx(indoor_processed, cv2.MORPH_CLOSE, kernel_indoor_v)
            
            candidates_indoor = find_plate_candidates_from_binary(indoor_processed, image)
            all_candidates.extend([(c[0], c[1], c[2], c[3], c[4], f"indoor_gamma_{gamma}") for c in candidates_indoor])
            
    except Exception as e:
        pass
    
    # Remove duplicates (similar positions)
    unique_candidates = []
    for candidate in all_candidates:
        x, y, w, h, area, method = candidate
        is_duplicate = False
        for existing in unique_candidates:
            ex, ey, ew, eh, ea, em = existing
            # Check if centers are close (within 20 pixels)
            if abs((x + w/2) - (ex + ew/2)) < 20 and abs((y + h/2) - (ey + eh/2)) < 20:
                is_duplicate = True
                # Keep the larger area
                if area > ea:
                    unique_candidates.remove(existing)
                    unique_candidates.append(candidate)
                break
        if not is_duplicate:
            unique_candidates.append(candidate)
    
    # Convert back to original format and sort
    final_candidates = [(x, y, w, h, area) for x, y, w, h, area, method in unique_candidates]
    final_candidates.sort(key=lambda x: x[4], reverse=True)
    return final_candidates[:10]
