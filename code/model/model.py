import cv2
import matplotlib.pyplot as plt
from preprocess import *
from segmentation import *
from feature_extraction import *
from postprocessing import *

def main():
    image_path = "F:\\France\\paris_cite\\S2\\image\\projet\\CoinDetector\\dataset\\images\\48.jpeg"
    original_image = cv2.imread(image_path)

    # Preprocess the image
    preprocessed_image = apply_gaussian_blur(original_image.copy())
    colored_preprocessed_image = apply_gamma_correction(preprocessed_image, gamma=1.5)
    preprocessed_image = convert_to_grayscale(colored_preprocessed_image.copy())
    preprocessed_image = apply_adaptive_histogram_equalization(preprocessed_image.copy())

    # Segment the image using otsu, adaptive, multi-otsu, and color-based thresholding
    otsu_segmented_image = apply_otsu_threshold(preprocessed_image.copy())
    multi_otsu_segmented_image, hist, peaks = multi_otsu_thresholding(preprocessed_image.copy())
    color_segmented_image = cv2.bitwise_not(color_based_segmentation(colored_preprocessed_image.copy()))
    adaptive_segmented_image = apply_adaptive_threshold(preprocessed_image.copy())



    # Apply Canny edge detection to the segmented images and post-process them
    cannied_image = apply_gaussian_blur(apply_closing(apply_canny_edge_detection(otsu_segmented_image.copy())))
    cannied_image1 = apply_gaussian_blur(apply_closing(apply_canny_edge_detection(multi_otsu_segmented_image.copy())))
    cannied_image2 = apply_gaussian_blur(apply_closing(apply_canny_edge_detection(color_segmented_image.copy())))
    cannied_image3 = apply_gaussian_blur(apply_closing(apply_canny_edge_detection(adaptive_segmented_image.copy())))


    # Find contours in the cannied images and display them
    contours = find_contours(cannied_image)
    image_with_contours = display_contours(preprocessed_image, contours)
    contours1 = find_contours(cannied_image1)
    image_with_contours1 = display_contours(preprocessed_image, contours1)
    contours2 = find_contours(cannied_image2)
    image_with_contours2 = display_contours(preprocessed_image, contours2)
    contours3 = find_contours(cannied_image3)
    image_with_contours3 = display_contours(preprocessed_image, contours3)


    # Apply Hough Circle Detection from contours and preprocessed image
    image_with_circles_preprocessed, circles = apply_hough_circle_detection_preprocessed(preprocessed_image.copy())

    #merged_contours = filter_contours([contours, contours1, contours2, contours3], circles)
    merged_contours = merge_and_postprocess_contours([contours, contours1, contours2, contours3], circles)

    image_with_merged_contours = display_contours(preprocessed_image, merged_contours)


    # Display all images side by side
    fig, axes = plt.subplots(4, 5, figsize=(18, 18))

    # Plot the original image
    axes[0, 0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    # Plot the otsu segmented image  
    axes[0, 1].imshow(otsu_segmented_image, cmap='gray')
    axes[0, 1].set_title('Otsu Segmented Image')
    axes[0, 1].axis('off')

    # Plot the multi-otsu segmented image
    axes[0, 2].imshow(multi_otsu_segmented_image, cmap='gray')
    axes[0, 2].set_title('Multi-Otsu Segmented Image')
    axes[0, 2].axis('off')

    # Plot the color-based segmented image
    axes[0, 3].imshow(color_segmented_image, cmap='gray')
    axes[0, 3].set_title('Color-based Segmented Image')
    axes[0, 3].axis('off')

    # Plot the adaptive segmented image
    axes[0, 4].imshow(adaptive_segmented_image, cmap='gray')
    axes[0, 4].set_title('Adaptive Segmented Image')
    axes[0, 4].axis('off')

    # Plot the preprocessed image
    axes[1, 0].imshow(preprocessed_image, cmap='gray')
    axes[1, 0].set_title('Preprocessed Image')
    axes[1, 0].axis('off')

    # Plot the otsu cannied image
    axes[1, 1].imshow(cannied_image, cmap='gray')
    axes[1, 1].set_title('Otsu Cannied Image')
    axes[1, 1].axis('off')

    # Plot the multi-otsu cannied image
    axes[1, 2].imshow(cannied_image1, cmap='gray')
    axes[1, 2].set_title('Multi-Otsu Cannied Image')
    axes[1, 2].axis('off')

    # Plot the color-based cannied image
    axes[1, 3].imshow(cannied_image2, cmap='gray')
    axes[1, 3].set_title('Color-based Cannied Image')
    axes[1, 3].axis('off')

    # Plot the adaptive cannied image
    axes[1, 4].imshow(cannied_image3, cmap='gray')
    axes[1, 4].set_title('Adaptive Cannied Image')
    axes[1, 4].axis('off')

    # Plot the image with Hough circles
    axes[2, 0].imshow(image_with_circles_preprocessed, cmap='gray')
    axes[2, 0].set_title('Hough Circles Image')
    axes[2, 0].axis('off')

    # Plot the contours from otsu cannied image
    axes[2, 1].imshow(image_with_contours, cmap='gray')
    axes[2, 1].set_title('Contours from Otsu Cannied')
    axes[2, 1].axis('off')

    # Plot the contours from multi-otsu cannied image
    axes[2, 2].imshow(image_with_contours1, cmap='gray')
    axes[2, 2].set_title('Contours from Multi-Otsu Cannied')
    axes[2, 2].axis('off')

    # Plot the contours from color-based cannied image
    axes[2, 3].imshow(image_with_contours2, cmap='gray')
    axes[2, 3].set_title('Contours from Color-based Cannied')
    axes[2, 3].axis('off')

    # Plot the contours from adaptive cannied image
    axes[2, 4].imshow(image_with_contours3, cmap='gray')
    axes[2, 4].set_title('Contours from Adaptive Cannied')
    axes[2, 4].axis('off')

    # Plot the histogram with peaks
    axes[3, 0].plot(hist)
    axes[3, 0].set_title('Histogram with Peaks')
    for peak in peaks:
        axes[3, 0].axvline(x=peak, color='r', linestyle='--')
    axes[3, 0].axis('off')

    # Plot the image with merged contours
    axes[3, 1].imshow(image_with_merged_contours, cmap='gray')
    axes[3, 1].set_title('Merged Contours')
    axes[3, 1].axis('off')

    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
