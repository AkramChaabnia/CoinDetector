import cv2

from model.feature_extraction import *


def empty(a):
    pass


def test_values(image):
    # Create trackbars for dp, minDis, param1, param2, minRadius and maxRadius control
    #dp=1.3, minDist=30, param1=150, param2=70, minRadius=78, maxRadius=0
    cv2.namedWindow('Settings', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Settings', 800, 250)
    cv2.createTrackbar('dp', 'Settings', 1, 10, empty)
    cv2.createTrackbar('minDist', 'Settings', 1, 100, empty)
    cv2.createTrackbar('param1', 'Settings', 1, 500, empty)
    cv2.createTrackbar('param2', 'Settings', 1, 100, empty)
    cv2.createTrackbar('minRadius', 'Settings', 1, 100, empty)
    cv2.createTrackbar('maxRadius', 'Settings', 1, 100, empty)

    while True:
        # Get current trackbar positions
        dp = cv2.getTrackbarPos('dp', 'Settings')
        minDist = cv2.getTrackbarPos('minDist', 'Settings')
        param1 = cv2.getTrackbarPos('param1', 'Settings')
        param2 = cv2.getTrackbarPos('param2', 'Settings')
        minRadius = cv2.getTrackbarPos('minRadius', 'Settings')
        maxRadius = cv2.getTrackbarPos('maxRadius', 'Settings')

        # Apply Hough Circle Transform to detect circles
        image_with_circles_preprocessed = apply_hough_circle_detection_preprocessed(image, cv2.HOUGH_GRADIENT, dp, minDist, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)
        
        # Check for key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()    
    return image_with_circles_preprocessed
