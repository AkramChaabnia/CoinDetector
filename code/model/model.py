import cv2
import matplotlib.pyplot as plt
from preprocess import *
from segmentation import *
from feature_extraction import *
from postprocessing import *

def main():
    image_path = "F:\\France\\paris_cite\\S2\\image\\projet\\CoinDetector\\dataset\\images\\40.jpg"
    original_image = cv2.imread(image_path)

    # Preprocess the image
    grayscale_image = convert_to_grayscale(image_path)
    processed_image = apply_gaussian_blur(grayscale_image)
    gamma_processed_image = apply_gamma_correction(processed_image, gamma=1.5)

    
    # Segment the image using color-based and Otsu thresholding
    otsu_segmented_image = apply_otsu_threshold(gamma_processed_image.copy())
    color_segmented_image = cv2.bitwise_not(color_based_segmentation(original_image))

    # Refine Otsu's segmentation using color-based segmentation
    refined_segmentation = combine_segmentation_results(otsu_segmented_image, color_segmented_image)
    refined_segmentation = apply_opening(apply_closing(refined_segmentation))

    # Apply Canny edge detection
    cannied_image = apply_canny_edge_detection(refined_segmentation)

    # Find contours in the cannied image and display them
    contours, hierarchy = find_contours(cannied_image)
    image_with_contours = display_contours(grayscale_image, contours)

    # Filter contours by circularity and aspect ratio and display them
    filtered_contours = filter_contours(contours, 0.5, 0.5)
    image_with_filtered_contours = display_contours(grayscale_image, filtered_contours)


    # Apply Hough Circle Detection from contours and preprocessed image
    image_with_circles_contours = apply_hough_circle_detection_contours(gamma_processed_image.copy(), filtered_contours)
    image_with_circles_preprocessed = apply_hough_circle_detection_preprocessed(gamma_processed_image.copy())

    # Display all images side by side
    fig, axes = plt.subplots(3, 4, figsize=(18, 10))

# Plot the original image
    axes[0, 0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

# Plot the grayscale image
    axes[0, 1].imshow(grayscale_image, cmap='gray')
    axes[0, 1].set_title('Grayscale Image')
    axes[0, 1].axis('off')

# Plot the otsu segmented image
    axes[0, 2].imshow(otsu_segmented_image, cmap='gray')
    axes[0, 2].set_title('Otsu Segmented Image')
    axes[0, 2].axis('off')

# Plot the color-based segmented image  
    axes[0, 3].imshow(color_segmented_image, cmap='gray')
    axes[0, 3].set_title('Color-based Segmentation')
    axes[0, 3].axis('off')

# Plot the refined segmentation
    axes[1, 0].imshow(refined_segmentation, cmap='gray')
    axes[1, 0].set_title('Combined Segmentation')
    axes[1, 0].axis('off')

# Plot the canny image
    axes[1, 1].imshow(cannied_image, cmap='gray')
    axes[1, 1].set_title('Canny Image')
    axes[1, 1].axis('off')

# Plot the contour image
    axes[1, 2].imshow(image_with_contours, cmap='gray')
    axes[1, 2].set_title('Contour Image')
    axes[1, 2].axis('off')

# Plot the filtered contours image
    axes[1, 3].imshow(image_with_filtered_contours, cmap='gray')
    axes[1, 3].set_title('Filtered Contours Image')
    axes[1, 3].axis('off')

# Plot the circles image from contours
    axes[2, 0].imshow(image_with_circles_contours, cmap='gray')
    axes[2, 0].set_title('Circles Image from Contours')
    axes[2, 0].axis('off')

# Plot the circles image from preprocessed image
    axes[2, 1].imshow(image_with_circles_preprocessed, cmap='gray')
    axes[2, 1].set_title('Circles Image from Preprocessed Image')
    axes[2, 1].axis('off')

# Plot the gamma processed image
    axes[2, 2].imshow(gamma_processed_image, cmap='gray')
    axes[2, 2].set_title('Gamma Processed Image')
    axes[2, 2].axis('off')


    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
