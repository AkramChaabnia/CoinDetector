# preprocess.py
import cv2
import numpy as np

def convert_to_grayscale(image_path):
    image = cv2.imread(image_path)
    
    if image is None:
        print("Error: Unable to load image")
        return None
    
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    return grayscale_image

def apply_gaussian_blur(image, kernel_size=(5, 5)):
    blurred_image = cv2.GaussianBlur(image, kernel_size, 0)
    
    return blurred_image

def apply_median_blur(image, kernel_size=5):
    blurred_image = cv2.medianBlur(image, kernel_size)
    
    return blurred_image

def apply_histogram_equalization(image):
    equalized_image = cv2.equalizeHist(image)
    
    return equalized_image

def apply_adaptive_histogram_equalization(image, clip_limit=2.0, tile_grid_size=(8, 8)):    
    # Create CLAHE object
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    
    # Apply CLAHE
    equalized_image = clahe.apply(image)
    
    return equalized_image


def apply_gamma_correction(image, gamma=1.0):
    # Apply gamma correction
    gamma_inv = 1.0 / gamma
    table = np.array([((i / 255.0) ** gamma_inv) * 255 for i in np.arange(0, 256)]).astype("uint8")
    corrected_image = cv2.LUT(image, table)
    return corrected_image


