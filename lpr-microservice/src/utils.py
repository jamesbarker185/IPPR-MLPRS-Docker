import cv2
import numpy as np
import base64

def identify_state(plate_text: str) -> str:
    """Identify Malaysian state from license plate text"""
    if not plate_text:
        return "Unknown"
    
    clean_plate = plate_text.replace(' ', '').upper()
    
    state_map = {
        # Single letter states
        "A": "Perak", "B": "Selangor", "C": "Pahang", "D": "Kelantan",
        "F": "W.P. Putrajaya", "J": "Johor", "K": "Kedah", "L": "W.P. Labuan",
        "M": "Melaka", "N": "Negeri Sembilan", "P": "Pulau Pinang", "Q": "Sarawak",
        "R": "Perlis", "S": "Sabah", "T": "Terengganu", "V": "W.P. Kuala Lumpur",
        "W": "W.P. Kuala Lumpur", "Z": "Military",
        
        # Multi-letter prefixes
        "KV": "Langkawi", "EV": "Special Series", "FFF": "Special Series",
        "VIP": "Special Series", "GOLD": "Special Series", "LIMO": "Special Series",
        "MADANI": "Special Series", "PETRA": "Special Series",
        "U": "Special Series", "X": "Special Series", "Y": "Special Series",
        "H": "Taxi"
    }
    
    # Check multi-letter prefixes first (longer matches)
    for prefix in sorted(state_map.keys(), key=len, reverse=True):
        if clean_plate.startswith(prefix):
            return state_map[prefix]
    
    # Default to unknown
    return "Unknown"

def enhance_plate_region(image: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
    """
    Apply multiple enhancement techniques to detected plate region
    
    Args:
        image: Input grayscale image
        x, y, w, h: Region of interest coordinates
        
    Returns:
        Enhanced plate region using multiple processing techniques
    """
    # Validate input coordinates
    if x < 0 or y < 0 or w <= 0 or h <= 0:
        return np.zeros((max(h, 50), max(w, 100)), dtype=np.uint8)
    
    # Extract region of interest with bounds checking
    height, width = image.shape[:2]
    
    # Ensure coordinates are within image bounds
    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))
    
    # Ensure width and height don't exceed image bounds
    x_end = min(x + w, width)
    y_end = min(y + h, height)
    
    # Recalculate actual width and height
    w = max(1, x_end - x)
    h = max(1, y_end - y)
    
    roi = image[y:y+h, x:x+w]
    
    # Handle empty or invalid ROI
    if roi.size == 0 or roi.shape[0] == 0 or roi.shape[1] == 0:
        # Return proportional fallback based on aspect ratio
        aspect_ratio = w / h if h > 0 else 2.0
        fallback_h = 50
        fallback_w = int(fallback_h * aspect_ratio)
        return np.zeros((fallback_h, fallback_w), dtype=np.uint8)
    
    # Start with a copy of the ROI
    enhanced_roi = roi.copy()
    
    # SMART enhancement: Different approaches based on region size
    try:
        region_area = enhanced_roi.shape[0] * enhanced_roi.shape[1]
        
        # MOTORCYCLE-SPECIFIC: For small regions (likely motorcycles), use different approach
        if region_area < 3000:  # Small motorcycle plate region
            
            # For small regions, upscale first to help OCR
            scale_factor = max(2, int(60 / min(enhanced_roi.shape[:2])))  # Scale to at least 60px height
            if scale_factor > 1:
                enhanced_roi = cv2.resize(enhanced_roi, 
                                        (enhanced_roi.shape[1] * scale_factor, 
                                         enhanced_roi.shape[0] * scale_factor), 
                                        interpolation=cv2.INTER_CUBIC)
            
            # More aggressive contrast enhancement for small text
            min_val, max_val = np.min(enhanced_roi), np.max(enhanced_roi)
            contrast_range = max_val - min_val
            
            if contrast_range < 150 and contrast_range > 0:  # More aggressive threshold for small plates
                enhanced_roi = ((enhanced_roi - min_val) * 255.0 / contrast_range).astype(np.uint8)
            
            # Light denoising for upscaled small images
            enhanced_roi = cv2.medianBlur(enhanced_roi, 3)
            
        else:  # Larger regions (likely cars) - use gentle enhancement
            
            # Method 1: Simple histogram stretching (very gentle)
            min_val, max_val = np.min(enhanced_roi), np.max(enhanced_roi)
            contrast_range = max_val - min_val
            
            # Only enhance if contrast is very poor (< 100 out of 255)
            if contrast_range < 100 and contrast_range > 0:
                enhanced_roi = ((enhanced_roi - min_val) * 255.0 / contrast_range).astype(np.uint8)
                pass
            else:
                pass
            
            # Method 2: Very light sharpening only for larger regions
            if enhanced_roi.shape[0] > 30 and enhanced_roi.shape[1] > 60:
                # Gentle unsharp mask
                gaussian = cv2.GaussianBlur(enhanced_roi, (3, 3), 1.0)
                enhanced_roi = cv2.addWeighted(enhanced_roi, 1.2, gaussian, -0.2, 0)
        
    except Exception as e:
        pass
        enhanced_roi = roi.copy()
    
    return enhanced_roi

def encode_image_base64(image_np: np.ndarray, format: str = '.jpg') -> str:
    """Encode numpy image to base64 string"""
    success, buffer = cv2.imencode(format, image_np)
    if not success:
        return ""
    return base64.b64encode(buffer).decode('utf-8')
