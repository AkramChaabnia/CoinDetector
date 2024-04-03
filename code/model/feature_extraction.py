# feature_extraction.py
import cv2
import numpy as np
from skimage import feature, transform
import numpy as np
from postprocessing import *


def apply_canny_edge_detection(image, threshold1=100, threshold2=200):
    # Apply Canny edge detection
    edges = cv2.Canny(image, threshold1, threshold2)
    
    return edges


#def detect_blobs(canny_image):
    # Create SimpleBlobDetector parameters
    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = 150
    params.maxThreshold = 230
    params.filterByArea = True
    params.minArea = 50
    params.maxArea = 50000
    params.filterByConvexity = True
    params.minConvexity = 0.9
    params.maxConvexity = 1.0

    params.filterByCircularity = False
    params.filterByInertia = False
    params.filterByColor = False

    # Create SimpleBlobDetector object
    detector = cv2.SimpleBlobDetector_create(params)

    # Detect keypoints (blobs) in the canny image
    keypoints = detector.detect(canny_image)

    # Draw keypoints on the canny image
    image_with_keypoints = cv2.drawKeypoints(canny_image, keypoints, None, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    return image_with_keypoints



#def calculate_circularity(contour):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return 0
    circularity = (4 * np.pi * area) / (perimeter * perimeter)
    return circularity

#def filter_contours(contours, max_aspect_ratio_deviation, min_circularity_threshold):
    filtered_contours = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h if h != 0 else 0
        circularity = calculate_circularity(contour)
        
        if abs(aspect_ratio - 1.0) <= max_aspect_ratio_deviation and circularity >= min_circularity_threshold:
            filtered_contours.append(contour)
    
    return filtered_contours

def apply_hough_circle_detection_contours(image, contours, dp=2, minDist=50, param1=200, param2=30, minRadius=20, maxRadius=100):
    # Create a mask to store all contours (single-channel)
    mask = np.zeros_like(image, dtype=np.uint8)

    # Draw contours on the mask
    cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)

    # Apply Hough Circle Transform to detect circles
    # You may need to adjust dp, minDist, param1, param2, minRadius, and maxRadius here
    circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, dp, minDist, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)

    # If circles are found, draw them on the original image
    if circles is not None:
        cimg = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # Convert to BGR for drawing
        circles = np.uint16(np.around(circles))
        for c in circles[0, :]:
            cv2.circle(cimg, (c[0], c[1]), c[2], (0, 255, 0), 3)  # Draw outer circle
            cv2.circle(cimg, (c[0], c[1]), 1, (0, 0, 255), 5)      # Draw center point
        return cimg
    return image

#def apply_hough_circle_detection_contours(image, contours, dp=2, minDist=20, param1=100, param2=20, minRadius=10, maxRadius=120):


#def def apply_hough_circle_detection_preprocessed(image, dp=1, minDist=50, param1=200, param2=30, minRadius=20, maxRadius=100):

#def apply_hough_circle_detection_preprocessed(image, dp=1.3, minDist=30, param1=150, param2=70, minRadius=78, maxRadius=0):
#def apply_hough_circle_detection_preprocessed(image, dp=1.3, minDist=30, param1=150, param2=70, minRadius=78, maxRadius=0):

#HoughCircles(image,coin,CV_HOUGH_GRADIENT,2,20,450,60,0,0 );

def apply_hough_circle_detection_preprocessed(image, dp=1.3, minDist=30, param1=150, param2=70, minRadius=78, maxRadius=0):

    # Apply Hough Circle Transform to detect circles
    circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, dp, minDist, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)

    # If circles are found, draw them on the original image
    if circles is not None:
        cimg = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # Convert to BGR for drawing
        circles = np.uint16(np.around(circles))
        for c in circles[0, :]:
            cv2.circle(cimg, (c[0], c[1]), c[2], (0, 255, 0), 3)  # Draw outer circle
            cv2.circle(cimg, (c[0], c[1]), 1, (0, 0, 255), 5)      # Draw center point
        return cimg, circles

    return image, None


