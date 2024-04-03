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

def find_contours_circles(image):
    # Find contours
    contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = check_circularity(contours, 0.3, 10000)
    circles = contours_to_circles(contours)
    return circles, contours, hierarchy


def display_contours(image, contours):
    # Draw contours on the original grayscale image
    image_with_contours = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(image_with_contours, contours, -1, (0, 255, 0), 2)
    return image_with_contours

def display_circles(image, circles):
    # Convert the image to BGR color
    image_with_circles = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Convert circles to integers
    circles = np.uint16(np.around(circles))

    # Draw each circle
    for c in circles:
        # Draw outer circle
        cv2.circle(image_with_circles, (c[0], c[1]), c[2], (0, 255, 0), 3)
        # Draw center point
        cv2.circle(image_with_circles, (c[0], c[1]), 1, (0, 0, 255), 5)

    return image_with_circles

def display_circles1(image, circles):
    image_with_circles = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for center, radius in circles:
        cv2.circle(image_with_circles, center, radius, (0, 255, 0), 2)

    """for contour in contours:
    # min area 
        if cv2.contourArea(contour) > 10000:
        # find radius
            (x,y),radius = cv2.minEnclosingCircle(contour) 
            center = (int(x),int(y))
            radius = int(radius)
            cv2.circle(image_with_contours,center,radius,(0,0,255),10)
            #coin_number+=1
"""
    return image_with_circles



def apply_hough_circle_detection_contours(image, contours, dp=2, minDist=50, param1=200, param2=30, minRadius=20, maxRadius=100):
    # Create a mask to store all contours (single-channel)
    mask = np.zeros_like(image, dtype=np.uint8)

    # Draw contours on the mask
    cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)

    # Preprocess the image
    # Apply Gaussian blur to reduce noise
    gray = cv2.GaussianBlur(image, (5, 5), 0)

    # Apply automatic Canny edge detection
    v = np.median(gray)
    sigma = 0.33
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edges = cv2.Canny(gray, lower, upper)

    # Apply Hough Circle Transform to detect circles
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp, minDist, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)

    # If circles are found, draw them on the original image
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for c in circles[0, :]:
            cv2.circle(image, (c[0], c[1]), c[2], (0, 255, 0), 3)  # Draw outer circle
            cv2.circle(image, (c[0], c[1]), 1, (0, 0, 255), 5)      # Draw center point
        return image, circles
    return image, None


def apply_hough_circle_detection_preprocessed(image, dp=1.3, minDist=85, param1=50, param2=40, minRadius=50, maxRadius=100):
    # Convert to grayscale if the image is in color
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Apply Hough Circle Transform to detect circles
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp, minDist, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)

    # If circles are found, draw them on the original image
    if circles is not None:
        cimg = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # Convert to BGR for drawing
        circles = np.uint16(np.around(circles))
        for c in circles[0, :]:
            cv2.circle(cimg, (c[0], c[1]), c[2], (0, 255, 0), 3)  # Draw outer circle
            cv2.circle(cimg, (c[0], c[1]), 1, (0, 0, 255), 5)      # Draw center point
        return cimg, circles

    return image, None