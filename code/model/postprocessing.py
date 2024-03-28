# postprocessing.py
import cv2

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

# Add more post-processing functions as needed
