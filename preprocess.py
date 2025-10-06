import cv2
import numpy as np

def adjust_gamma(image, gamma=1.0):
    """Gamma correction to normalize lighting."""
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def clahe_enhancement(image):
    """Contrast Limited Adaptive Histogram Equalization (CLAHE) for low-light images."""
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge((l_clahe, a, b))
    return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)

def preprocess_custom_features(image):
    """Updated with lighting normalization."""
    # Gamma correction (adjust gamma based on lighting)
    gamma = 0.5 if np.mean(image) < 100 else 1.5  
    image = adjust_gamma(image, gamma=gamma)
    
    # CLAHE for contrast enhancement
    image = clahe_enhancement(image)
    
    # Original feature extraction (edges + black spots)
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if gray_image.dtype != np.uint8:
        gray_image = (gray_image * 255).astype(np.uint8)
    
    edges = cv2.Canny(gray_image, threshold1=100, threshold2=200)
    _, black_spots = cv2.threshold(gray_image, 50, 255, cv2.THRESH_BINARY_INV)
    
    return np.stack([gray_image, edges, black_spots], axis=-1)
