# postprocessing.py
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import heapq



def apply_opening(image, kernel_size=3, iterations=1):
    """
    Apply opening to the input image.

    Parameters:
        image (numpy.ndarray): Input image.
        kernel_size (int): Size of the structuring element.
        iterations (int): Number of times opening is applied.

    Returns:
        numpy.ndarray: Image after opening operation.
    """
    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    opened_image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=iterations)
    return opened_image

def apply_closing(image, kernel_size=3, iterations=1):
    """
    Apply closing to the input image.

    Parameters:
        image (numpy.ndarray): Input image.
        kernel_size (int): Size of the structuring element.
        iterations (int): Number of times closing is applied.

    Returns:
        numpy.ndarray: Image after closing operation.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    closed_image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    return closed_image

def apply_morphological_gradient(image, kernel_size=3):
    """
    Apply morphological gradient operation to the input image.

    Parameters:
        image (numpy.ndarray): Input image.
        kernel_size (int): Size of the structuring element.

    Returns:
        numpy.ndarray: Image after morphological gradient operation.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    gradient_image = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
    return gradient_image


def apply_erosion(image, kernel_size=3, iterations=1):
    """
    Apply erosion to the input image.

    Parameters:
        image (numpy.ndarray): Input image.
        kernel_size (int): Size of the structuring element.
        iterations (int): Number of times erosion is applied.

    Returns:
        numpy.ndarray: Eroded image.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    eroded_image = cv2.erode(image, kernel, iterations=iterations)
    return eroded_image

def apply_dilation(image, kernel_size=3, iterations=1):
    """
    Apply dilation to the input image.

    Parameters:
        image (numpy.ndarray): Input image.
        kernel_size (int): Size of the structuring element.
        iterations (int): Number of times dilation is applied.

    Returns:
        numpy.ndarray: Dilated image.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    dilated_image = cv2.dilate(image, kernel, iterations=iterations)
    return dilated_image

def non_max_suppression(contours, min_area_threshold=1400, overlapThresh=0.5):
    if len(contours) == 0:
        return []

    pick = []

    # Calculate the bounding boxes for all contours
    boxes = [cv2.boundingRect(cnt) for cnt in contours]

    # Convert the bounding boxes to a NumPy array
    boxes = np.array(boxes)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]

    area = boxes[:, 2] * boxes[:, 3]
    idxs = np.argsort(y2)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]

        # Check if the area of the contour is above the threshold
        if area[i] >= min_area_threshold:
            pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs[:last]]

        # Find contours with overlap greater than the threshold
        suppress = np.concatenate(([last], np.where(overlap > overlapThresh)[0]))

        # Delete the suppressed contours from idxs
        idxs = np.delete(idxs, suppress)

    # Convert the pick indices to contours
    picked_contours = [contours[i] for i in pick]

    return picked_contours


def cluster_contours(contours):
    # Flatten the list of contours to a 2D array
    flattened_contours = np.concatenate([contour.squeeze() for contour in contours])

    # Apply DBSCAN
    db = DBSCAN(eps=3, min_samples=2).fit(flattened_contours)

    # Get the labels of the clusters
    labels = db.labels_

    # Create a list to hold the clustered contours
    clustered_contours = []

    # Group the contours by their cluster label
    for label in set(labels):
        if label == -1:
            # Ignore noise (contours not in any cluster)
            continue

        # Get the contours in this cluster
        cluster = flattened_contours[labels == label]

        # Add the cluster to the list
        clustered_contours.append(cluster)

    return clustered_contours

#def merge_and_postprocess_contours(contours_list, circles):
    # Merge contours into a single list
    merged_contours = [cnt for sublist in contours_list for cnt in sublist]

    # Add circular contours to the merged contours
    if circles is not None:
        for circle in circles[0]:
            center = (circle[0], circle[1])
            radius = circle[2]
            # Create a list to hold the points on the circle
            circle_contour = []
            # Define theta within the loop
            for theta in np.linspace(0, 2*np.pi, 50):
                point = np.array([center[0] + radius * np.cos(theta), center[1] + radius * np.sin(theta)], dtype=np.int32)
                circle_contour.append(point)
            # Convert the list to a 2D array and append it to merged_contours
            merged_contours.append(np.array(circle_contour))


    # Apply NMS to the contours
    merged_contours = non_max_suppression(merged_contours)
    # Cluster the circular contours
    #merged_contours = cluster_contours(merged_contours)

    return merged_contours

#def merge_and_postprocess_contours1(contours_list, circles):
    # Merge contours into a single list
    merged_contours = [cnt for sublist in contours_list for cnt in sublist]

    # Add circles to the merged contours
    if circles is not None:
        for circle in circles[0]:
            center = (circle[0], circle[1])
            radius = circle[2]
            # Define theta within the loop
            for theta in np.linspace(0, 2*np.pi, 50):
                circle_contour = np.array([[center[0] + radius * np.cos(theta)], [center[1] + radius * np.sin(theta)]], dtype=np.int32).T
                merged_contours.append(circle_contour)

    # Apply post-processing techniques
    merged_contours = non_max_suppression(merged_contours)
    merged_contours = cluster_contours(merged_contours)

    return merged_contours


def check_circularity(contours, min_circularity, min_area_threshold):
    circular_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue

        circularity = 4 * np.pi * area / (perimeter ** 2)
        if circularity >= min_circularity and area >= min_area_threshold:
            circular_contours.append(contour)
    return circular_contours

def contours_to_circles(contours):
    circles = []
    for contour in contours:
        # min area 
        if cv2.contourArea(contour) > 10000:
            # find radius
            (x, y), radius = cv2.minEnclosingCircle(contour)
            circles.append([int(x), int(y), int(radius)])
    return np.array(circles)

def merge_and_postprocess_circles(circles_list, overlapThresh=0.5):
    merged_circles = []
    for sublist in circles_list:
        for circle in sublist:
            if len(circle) == 3:  # Ensure the circle is represented as (x, y, radius)
                merged_circles.append(circle)
    merged_circles = non_max_suppression_circles(merged_circles, overlapThresh)
    return merged_circles

def non_max_suppression_circles(circles, overlapThresh, size_ratio=0.5, multi_overlap_thresh=2, distance_thresh=20):
    if len(circles) == 0:
        return []

    # Pre-processing step to remove circles that contain more than two other circles
    circles = [(circle[2], tuple(circle)) for circle in circles]
    circles_to_remove = []
    for i, (_, circle) in enumerate(circles):
        count = sum(1 for _, other_circle in circles if np.sqrt((other_circle[0] - circle[0])**2 + (other_circle[1] - circle[1])**2) + other_circle[2] < circle[2])
        if count > multi_overlap_thresh:
            circles_to_remove.append(i)
    for i in sorted(circles_to_remove, reverse=True):
        del circles[i]

    # Continue with the rest of the non-maximum suppression algorithm
    pick = []
    heapq.heapify(circles)

    while circles:
        _, circle = heapq.heappop(circles)
        pick.append(circle)

        suppress = []

        for i in range(len(circles)):
            _, other_circle = circles[i]
            if np.sqrt((other_circle[0] - circle[0])**2 + (other_circle[1] - circle[1])**2) + min(other_circle[2], circle[2]) < max(other_circle[2], circle[2]):
                # If the smaller circle is significantly smaller than the larger one, suppress the smaller one
                if min(other_circle[2], circle[2]) / max(other_circle[2], circle[2]) < size_ratio:
                    if other_circle[2] < circle[2]:
                        suppress.append(i)
                    else:
                        pick.remove(circle)
                        break
                else:
                    suppress.append(i)

        # Remove circles from the end of the list to avoid changing the indices of the remaining circles
        for i in sorted(suppress, reverse=True):
            del circles[i]

    # Remove duplicates
    unique_circles = []
    for circle1 in pick:
        is_duplicate = False
        for circle2 in unique_circles:
            distance = np.sqrt((circle1[0] - circle2[0])**2 + (circle1[1] - circle2[1])**2)
            if distance < distance_thresh:
                is_duplicate = True
                if circle1[2] > circle2[2]:  # If circle1 has a larger radius, replace circle2 with circle1
                    unique_circles.remove(circle2)
                    unique_circles.append(circle1)
                break
        if not is_duplicate:
            unique_circles.append(circle1)

    return unique_circles
