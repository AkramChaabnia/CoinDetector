import cv2
import matplotlib.pyplot as plt
from preprocess import *
from segmentation import *
from feature_extraction import *
from postprocessing import *

def main():
    image_path = "F:\\France\\paris_cite\\S2\\image\\projet\\CoinDetector\\dataset\\images\\60.jpg"
    original_image = cv2.imread(image_path)

    # Preprocess the image
    processed_image = apply_gaussian_blur(original_image.copy())
    gamma_processed_image = apply_gamma_correction(processed_image, gamma=1.5)
    grayscale_image = convert_to_grayscale(gamma_processed_image.copy())
    hist_processed_image = apply_adaptive_histogram_equalization(grayscale_image.copy())


    # Segment the image using color-based and Otsu thresholding
    otsu_segmented_image = apply_otsu_threshold(hist_processed_image.copy())
    adaptive_segmented_image = apply_adaptive_threshold(hist_processed_image.copy())
    multi_otsu_segmented_image, hist, peaks = multi_otsu_thresholding(hist_processed_image.copy())

    #color_segmented_image = cv2.bitwise_not(color_based_segmentation(original_image.copy()))

    # Refine Otsu's segmentation using color-based segmentation
    #refined_segmentation = combine_segmentation_results(otsu_segmented_image, color_segmented_image)
    #refined_segmentation = apply_opening(apply_closing(refined_segmentation))

    # Apply Canny edge detection to the grayscale image
    cannied_image = apply_canny_edge_detection(otsu_segmented_image.copy())
    cannied_image1 = apply_canny_edge_detection(adaptive_segmented_image.copy())
    cannied_image2 = apply_canny_edge_detection(multi_otsu_segmented_image.copy())


    # Find contours in the cannied image and display them
    contours = find_contours(cannied_image)
    image_with_contours = display_contours(hist_processed_image, contours)
    # Find contours in the cannied image and display them
    contours1 = find_contours(cannied_image1)
    image_with_contours1 = display_contours(hist_processed_image, contours)
    # Find contours in the cannied image and display them
    contours2 = find_contours(cannied_image2)
    image_with_contours2 = display_contours(hist_processed_image, contours)

    # Filter contours by circularity and aspect ratio and display them
    filtered_contours = filter_contours(contours, 0.5, 0.5)
    image_with_filtered_contours = display_contours(hist_processed_image, filtered_contours)


    # Apply Hough Circle Detection from contours and preprocessed image
    image_with_circles_contours = apply_hough_circle_detection_contours(hist_processed_image.copy(), filtered_contours)
    image_with_circles_preprocessed = apply_hough_circle_detection_preprocessed(hist_processed_image.copy())

    # Display all images side by side
    fig, axes = plt.subplots(3, 4, figsize=(18, 10))

# Plot the original image
    axes[0, 0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

# Plot the otsu segmented image
    axes[0, 1].imshow(otsu_segmented_image, cmap='gray')
    axes[0, 1].set_title('Otsu Segmented Image')
    axes[0, 1].axis('off')

# Plot the adaptive segmented image  
    axes[0, 2].imshow(adaptive_segmented_image, cmap='gray')
    axes[0, 2].set_title('Color-based Segmentation')
    axes[0, 2].axis('off')

# Plot the multi otsu segmentation
    axes[0, 3].imshow(multi_otsu_segmented_image, cmap='gray')
    axes[0, 3].set_title('multi otsu Segmented Image')
    axes[0, 3].axis('off')

    # Plot the grayscale image
    axes[1, 0].imshow(hist_processed_image, cmap='gray')
    axes[1, 0].set_title('histogram equalized Image')
    axes[1, 0].axis('off')


# Plot the otsu cannied image
    axes[1, 1].imshow(cannied_image, cmap='gray')
    axes[1, 1].set_title('otsu cannied image')
    axes[1, 1].axis('off')

# Plot the adaptive cannied image
    axes[1, 2].imshow(cannied_image1, cmap='gray')
    axes[1, 2].set_title('adaptive cannied image')
    axes[1, 2].axis('off')

# Plot the multi otsu cannied image
    axes[1, 3].imshow(cannied_image2, cmap='gray')
    axes[1, 3].set_title('multi otsu cannied image')
    axes[1, 3].axis('off')

# Plot the histogram with peaks
    axes[2, 0].plot(hist)
    axes[2, 0].set_title('Histogram with Peaks')
    for peak in peaks:
        axes[2, 0].axvline(x=peak, color='r', linestyle='--')

# Plot the circles from otsu cannied image
    axes[2, 1].imshow(image_with_contours, cmap='gray')
    axes[2, 1].set_title('contours from otsu cannied image')
    axes[2, 1].axis('off')

# Plot the circles from adaptive cannied image
    axes[2, 2].imshow(image_with_contours1)
    axes[2, 2].set_title('contours from adaptive cannied image')
    axes[2, 2].axis('off')

# Plot the circles from multi otsu cannied image
    axes[2, 3].imshow(image_with_contours2, cmap='gray')
    axes[2, 3].set_title('contours from multi otsu cannied image')
    axes[2, 3].axis('off')




    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
