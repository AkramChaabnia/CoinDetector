# preprocess.py
import cv2
import numpy as np

def convert_to_grayscale(image):  
    if image is None:
        print("Error: Unable to load image")
        return None
    
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    return grayscale_image

def apply_gaussian_blur(image, kernel_size=(7, 7)):
    blurred_image = cv2.GaussianBlur(image, kernel_size, 3)
    
    return blurred_image

#gaussian blur with 7*7 kernel and 1.5 sigma as blur
def apply_median_blur(image, kernel_size=5):
    blurred_image = cv2.medianBlur(image, kernel_size)
    
    return blurred_image

def apply_gamma_correction(image, gamma=1.0):
    # Apply gamma correction
    gamma_inv = 1.0 / gamma
    table = np.array([((i / 255.0) ** gamma_inv) * 255 for i in np.arange(0, 256)]).astype("uint8")
    corrected_image = cv2.LUT(image, table)
    return corrected_image


def apply_histogram_equalization(image):
    # Check if the image is not grayscale
    if len(image.shape) > 2 and image.shape[2] > 1:
        # Convert the image to grayscale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply histogram equalization
    equalized_image = cv2.equalizeHist(image)

    return equalized_image



def apply_adaptive_histogram_equalization(image, clip_limit=2.0, tile_grid_size=(14, 14)):    
    # Create CLAHE object
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    # Check if the image is not grayscale
    if len(image.shape) > 2 and image.shape[2] > 1:
        # Convert the image to grayscale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply CLAHE to the grayscale image
    clahe_image = clahe.apply(image)

    return clahe_image



