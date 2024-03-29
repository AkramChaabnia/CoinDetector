# feature_extraction.py
import cv2
import numpy as np
from skimage import feature, transform
import numpy as np


def apply_canny_edge_detection(image, threshold1=100, threshold2=200):
    # Apply Canny edge detection
    edges = cv2.Canny(image, threshold1, threshold2)
    
    return edges

def find_contours(image):
    # Find contours
    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours, hierarchy


def calculate_circularity(contour):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    circularity = (4 * np.pi * area) / (perimeter * perimeter)
    return circularity

def filter_contours(contours, max_aspect_ratio_deviation, min_circularity_threshold):
    filtered_contours = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h if h != 0 else 0
        circularity = calculate_circularity(contour)
        
        if abs(aspect_ratio - 1.0) <= max_aspect_ratio_deviation and circularity >= min_circularity_threshold:
            filtered_contours.append(contour)
    
    return filtered_contours


def display_contours(grayscale_image, contours):
    # Draw contours on the original grayscale image
    image_with_contours = cv2.cvtColor(grayscale_image, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(image_with_contours, contours, -1, (0, 255, 0), 2)
    
    return image_with_contours

