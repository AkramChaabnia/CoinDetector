# postprocessing.py
import cv2
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import numpy as np

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
    # Convert contours to a list of arrays, each representing a contour
    contour_arrays = [np.squeeze(cnt) for cnt in contours]

    # Flatten the list of contour arrays to a 2D array
    flattened_contours = np.vstack(contour_arrays)

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

        # Get the indices of contours in this cluster
        indices = np.where(labels == label)[0]

        # Ensure that the indices are within the range of the contours tuple
        indices = indices[indices < len(contours)]

        # Get the contours in this cluster
        cluster = [contours[i] for i in indices]

        # Flatten the list of contours in the cluster
        flattened_cluster = [np.squeeze(cnt) for cnt in cluster]

        # Add the flattened cluster to the list
        clustered_contours.extend(flattened_cluster)

    return clustered_contours

def merge_and_postprocess_contours(contours_list, circles):
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

