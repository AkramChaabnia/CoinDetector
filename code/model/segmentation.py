# segmentation.py
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_multiotsu
from scipy.signal import find_peaks
from .postprocessing import *
from .preprocess import *


def apply_segmentation(hist_processed_image, thresholds):
    # Apply thresholding
    segmented_image = np.zeros_like(hist_processed_image)
    for threshold in thresholds:
        segmented_image[hist_processed_image > threshold] = 255

    return segmented_image

def apply_segmentation1(hist_processed_image, thresholds):
    # Appliquer la segmentation basée sur les seuils
    #_, segmented_image = cv2.threshold(hist_processed_image, thresholds[0], 255, cv2.THRESH_BINARY)
    segmented_image = np.digitize(hist_processed_image, bins=thresholds)

    return segmented_image

def apply_segmentation2(hist_processed_image, thresholds):
    # Initialize segmented image
    segmented_image = np.zeros_like(hist_processed_image)

    # Assign different intensity values to different classes based on thresholds
    num_classes = len(thresholds) + 1
    for i, threshold in enumerate(thresholds):
        intensity = int(255 * (i+1) / num_classes)  # Change here
        if i == 0:
            # Pixels below the first threshold belong to class 0
            segmented_image[hist_processed_image <= threshold] = intensity
        elif i == len(thresholds):
            # Pixels above the last threshold belong to the last class
            segmented_image[hist_processed_image > thresholds[-1]] = 255
        else:
            # Pixels between consecutive thresholds belong to intermediate classes
            segmented_image[(hist_processed_image > thresholds[i-1]) & (hist_processed_image <= threshold)] = intensity

    # Invert segmented image to have white regions
    inverted_segmented_image = 255 - segmented_image

    return inverted_segmented_image


def apply_otsu_threshold(image):
    # Apply Otsu's thresholding
    _, segmented_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Apply morphological operations to enhance the segmentation
    segmented_image = apply_opening(segmented_image)
    # Apply other post-processing operations as needed
    segmented_image = apply_gaussian_blur(apply_median_blur(segmented_image))
    
    return segmented_image


def apply_adaptive_threshold(image):
    # Apply adaptive thresholding
    segmented_image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 7)
    #segmented_image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    return segmented_image

def multi_otsu_thresholding(image):
    # Calculer l'histogramme
    hist = np.histogram(image.ravel(), bins=256)[0]
    
    # Détecter les pics significatifs dans l'histogramme
    peaks, _ = find_peaks(hist, height=5000, width=3, distance=60)
    print("Pics Détectés:", peaks)

    # Déterminer le nombre de classes pour la segmentation
    num_classes = len(peaks) + 1  # Ajouter 1 car le nombre de classes est le nombre de seuils + 1
    print("Nombre de Classes pour la Segmentation:", num_classes)

    # Appliquer la segmentation avec multi-Otsu
    thresholds = threshold_multiotsu(image, classes=num_classes)
    print("Seuils de Segmentation:", thresholds)

    # Appliquer la segmentation
    segmented_image = apply_segmentation(image, thresholds)

    return segmented_image, hist, peaks

def color_based_segmentation(image):
    # Convert image from BGR to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Initialize segmented image
    segmented_image = np.zeros_like(image)

    # Apply adaptive thresholding on the saturation channel to adapt to lighting conditions
    saturation_channel = hsv_image[:, :, 1]
    adaptive_threshold = apply_adaptive_threshold(saturation_channel)

    # Define color ranges for euro coins
    color_ranges = {
        "1_cent": [(0, 100, 100), (10, 255, 255)],    # Reddish color
        "2_cent": [(0, 100, 100), (10, 255, 255)],    # Reddish color
        "5_cent": [(0, 100, 100), (10, 255, 255)],    # Reddish color
        "10_cent": [(0, 100, 100), (20, 255, 255)],   # Reddish color
        "20_cent": [(0, 100, 100), (20, 255, 255)],   # Reddish color
        "50_cent": [(0, 100, 100), (20, 255, 255)],   # Reddish color
        "1_euro": [(20, 100, 100), (35, 255, 255)],  # Gold color
        "2_euro": [(20, 100, 100), (35, 255, 255)],   # Gold color
        "1_euros": [(10, 100, 100), (25, 255, 255)],  # Adjusted for both gold and silver colors
        "2_euros": [(10, 100, 100), (25, 255, 255)]   # Adjusted for both gold and silver colors

    }

    # Apply color-based segmentation for each coin denomination
    for coin, (lower, upper) in color_ranges.items():
        print("Segmenting", coin, "with HSV range:", lower, "-", upper)
        # Threshold the HSV image to get only pixels within the specified color range
        mask = cv2.inRange(hsv_image, np.array(lower), np.array(upper))
        
        # Apply the adaptive thresholding mask to refine the segmentation
        mask = cv2.bitwise_and(mask, adaptive_threshold)

        # Apply the mask to the original image
        coin_segment = cv2.bitwise_and(image, image, mask=mask)
        
        # Add the segmented coin to the segmented image
        segmented_image = cv2.add(segmented_image, coin_segment)

    # Convert the color-based segmented image to grayscale
    segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)
    # Apply morphological closing to fill in any gaps in the segmented regions
    segmented_image = apply_closing(segmented_image)
    # Apply other post-processing operations as needed
    segmented_image = apply_gaussian_blur(apply_median_blur(segmented_image))

    return segmented_image

def edge_based_segmentation(image):
    # Convert the image to grayscale
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Canny edge detection
    edges = cv2.Canny(grayscale_image, 30, 100)  # Adjust thresholds as needed

    # Apply morphological operations to enhance edges
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.erode(edges, kernel, iterations=1)

    # Find contours in the edge-detected image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a black background image
    segmented_image = np.zeros_like(image)
    
    # Draw contours on the segmented image
    cv2.drawContours(segmented_image, contours, -1, (255, 255, 255), thickness=cv2.FILLED)  # Fills contours with white
    
    return segmented_image

def combine_segmentation_results(seg_mask1, seg_mask2):
    # Calculate overlapping regions
    overlapping_regions = cv2.bitwise_and(seg_mask1, seg_mask2)

    # Calculate area of each segmentation mask and overlapping region
    area_mask1 = np.sum(seg_mask1 / 255)
    area_mask2 = np.sum(seg_mask2 / 255)
    area_overlapping = np.sum(overlapping_regions / 255)

    # Calculate confidence scores
    confidence_score1 = area_overlapping / area_mask1 if area_mask1 != 0 else 0
    confidence_score2 = area_overlapping / area_mask2 if area_mask2 != 0 else 0

    # Normalize confidence scores
    total_confidence = confidence_score1 + confidence_score2
    if total_confidence != 0:
        confidence_score1 /= total_confidence
        confidence_score2 /= total_confidence

    # Combine segmentation masks using weighted sum
    combined_mask = (seg_mask1 * confidence_score1) + (seg_mask2 * confidence_score2)

    # Apply thresholding and post-processing as needed
    _, combined_mask = cv2.threshold(combined_mask, 127, 255, cv2.THRESH_BINARY)

# Apply morphological operations for further refinement
    kernel = np.ones((5, 5), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)


    return combined_mask.astype(np.uint8)

