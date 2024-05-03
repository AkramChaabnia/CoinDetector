import cv2
import matplotlib.pyplot as plt
from .preprocess import *
from .segmentation import *
from .feature_extraction import *
from .postprocessing import *

def model_test(image_path, display=False):
    #image_path = "dataset\reste\203.jpg"
    original_image = cv2.imread(image_path)
        # Check if the image is loaded correctly
    if original_image is None:
        print(f"Failed to load image at {image_path}")
        return


    # Preprocess the image
    preprocessed_image = apply_gaussian_blur(original_image.copy())
    preprocessed_image1 = apply_gamma_correction(preprocessed_image.copy())
    preprocessed_image2 = convert_to_grayscale(preprocessed_image1.copy())
    preprocessed_image3 = apply_median_blur(preprocessed_image2.copy())
    preprocessed_image4 = apply_adaptive_histogram_equalization(preprocessed_image3.copy())

    # Segment the image using otsu, adaptive, multi-otsu, and color-based thresholding
    otsu_segmented_image = apply_otsu_threshold(preprocessed_image4.copy())
    multi_otsu_segmented_image, hist, peaks = multi_otsu_thresholding(preprocessed_image4.copy())
    color_segmented_image = color_based_segmentation(preprocessed_image.copy())
    adaptive_segmented_image = apply_adaptive_threshold(preprocessed_image3.copy())



    # Apply Canny edge detection to the segmented images and post-process them
    cannied_image = apply_gaussian_blur(apply_closing(apply_canny_edge_detection(otsu_segmented_image.copy())))
    cannied_image1 = apply_gaussian_blur(apply_closing(apply_canny_edge_detection(multi_otsu_segmented_image.copy())))
    cannied_image2 = apply_gaussian_blur(apply_closing(apply_canny_edge_detection(color_segmented_image.copy())))
    cannied_image3 = apply_gaussian_blur(apply_closing(apply_canny_edge_detection(adaptive_segmented_image.copy())))


    # Find contours in the cannied images and display them
    circles, contours, hierarchy = find_contours_circles(cannied_image)
    image_with_contours = display_contours(preprocessed_image4, contours)
    image_with_circles, number_of_coins = display_circles(preprocessed_image4, circles)

    circles1, contours1, hierarchy1 = find_contours_circles(cannied_image1)
    image_with_contours1 = display_contours(preprocessed_image4, contours1)
    image_with_circles1, number_of_coins1 = display_circles(preprocessed_image4, circles1)

    circles2, contours2, hierarchy2 = find_contours_circles(cannied_image2)
    image_with_contours2 = display_contours(preprocessed_image4, contours2)
    image_with_circles2, number_of_coins2 = display_circles(preprocessed_image4, circles2)

    circles3, contours3, hierarchy3 = find_contours_circles(cannied_image3)
    image_with_contours3 = display_contours(preprocessed_image4, contours3)
    image_with_circles3, number_of_coins3 = display_circles(preprocessed_image4, circles3)

    # Apply Hough Circle Detection from contours and preprocessed image
    #image_with_circles_preprocessed, circle = apply_hough_circle_detection_preprocessed(preprocessed_image3.copy())

    merged_contours = merge_and_postprocess_circles([circles, circles1, circles2, circles3])
    image_with_merged_contours, number_of_coin = display_circles(preprocessed_image4, merged_contours)

    if display==True:
    # Display all images side by side
        fig, axes = plt.subplots(5, 5, figsize=(18, 18))

    # Plot the original image
        axes[0, 0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Original Image', color='blue', fontsize=10)
        axes[0, 0].axis('off')

    # Plot the otsu segmented image  
        axes[0, 1].imshow(otsu_segmented_image, cmap='gray')
        axes[0, 1].set_title('Otsu Segmented Image', color='blue', fontsize=10)
        axes[0, 1].axis('off')

    # Plot the multi-otsu segmented image
        axes[0, 2].imshow(multi_otsu_segmented_image, cmap='gray')
        axes[0, 2].set_title('Multi-Otsu Segmented Image', color='blue', fontsize=10)
        axes[0, 2].axis('off')

    # Plot the color-based segmented image
        axes[0, 3].imshow(color_segmented_image, cmap='gray')
        axes[0, 3].set_title('Color-based Segmented Image', color='blue', fontsize=10)
        axes[0, 3].axis('off')

    # Plot the adaptive segmented image
        axes[0, 4].imshow(adaptive_segmented_image, cmap='gray')
        axes[0, 4].set_title('Adaptive Segmented Image', color='blue', fontsize=10)
        axes[0, 4].axis('off')

    # Plot the preprocessed image
    #axes[1, 0].imshow(preprocessed_image, cmap='gray')
        axes[1, 0].imshow(cv2.cvtColor(preprocessed_image1, cv2.COLOR_BGR2RGB))
        axes[1, 0].set_title('Preprocessed Image', color='blue', fontsize=10)
        axes[1, 0].axis('off')

    # Plot the otsu cannied image
        axes[1, 1].imshow(cannied_image, cmap='gray')
        axes[1, 1].set_title('Otsu Cannied Image', color='blue', fontsize=10)
        axes[1, 1].axis('off')

    # Plot the multi-otsu cannied image
        axes[1, 2].imshow(cannied_image1, cmap='gray')
        axes[1, 2].set_title('Multi-Otsu Cannied Image', color='blue', fontsize=10)
        axes[1, 2].axis('off')

    # Plot the color-based cannied image
        axes[1, 3].imshow(cannied_image2, cmap='gray')
        axes[1, 3].set_title('Color-based Cannied Image', color='blue', fontsize=10)
        axes[1, 3].axis('off')

    # Plot the adaptive cannied image
        axes[1, 4].imshow(cannied_image3, cmap='gray')
        axes[1, 4].set_title('Adaptive Cannied Image', color='blue', fontsize=10)
        axes[1, 4].axis('off')

    # Plot the image with Hough circles
        #axes[2, 0].imshow(image_with_circles_preprocessed, cmap='gray')
        #axes[2, 0].set_title('Hough Circles Image', color='blue', fontsize=10)
        #axes[2, 0].axis('off')

    # Plot the contours from otsu cannied image
        axes[2, 1].imshow(image_with_contours, cmap='gray')
        axes[2, 1].set_title('Contours from Otsu Cannied', color='blue', fontsize=10)
        axes[2, 1].axis('off')

    # Plot the contours from multi-otsu cannied image
        axes[2, 2].imshow(image_with_contours1, cmap='gray')
        axes[2, 2].set_title('Contours from Multi-Otsu Cannied', color='blue', fontsize=10)
        axes[2, 2].axis('off')

    # Plot the contours from color-based cannied image
        axes[2, 3].imshow(image_with_contours2, cmap='gray')
        axes[2, 3].set_title('Contours from Color-based Cannied', color='blue', fontsize=10)
        axes[2, 3].axis('off')

    # Plot the contours from adaptive cannied image
        axes[2, 4].imshow(image_with_contours3, cmap='gray')
        axes[2, 4].set_title('Contours from Adaptive Cannied', color='blue', fontsize=10)
        axes[2, 4].axis('off')

    # Plot the histogram with peaks
        axes[3, 0].plot(hist)
        axes[3, 0].set_title('Histogram with Peaks', color='blue', fontsize=10)
        for peak in peaks:
            axes[3, 0].axvline(x=peak, color='r', linestyle='--')
        axes[3, 0].axis('off')

    # Plot the circles from otsu cannied image
        axes[3, 1].imshow(image_with_circles, cmap='gray')
        axes[3, 1].text(10, 30, 'Total Coins: ' + str(number_of_coins), color='red', fontsize=12)
        axes[3, 1].set_title('Circles from Otsu Cannied', color='blue', fontsize=10)
        axes[3, 1].axis('off')

    # Plot the circles from multi-otsu cannied image
        axes[3, 2].imshow(image_with_circles1, cmap='gray')
        axes[3, 2].text(10, 30, 'Total Coins: ' + str(number_of_coins1), color='red', fontsize=12)
        axes[3, 2].set_title('circles from Multi-Otsu Cannied', color='blue', fontsize=10)
        axes[3, 2].axis('off')

    # Plot the circles from color-based cannied image
        axes[3, 3].imshow(image_with_circles2, cmap='gray')
        axes[3, 3].text(10, 30, 'Total Coins: ' + str(number_of_coins2), color='red', fontsize=12)
        axes[3, 3].set_title('circles from Color-based Cannied', color='blue', fontsize=10)
        axes[3, 3].axis('off')

    # Plot the circles from adaptive cannied image
        axes[3, 4].imshow(image_with_circles3, cmap='gray')
        axes[3, 4].text(10, 30, 'Total Coins: ' + str(number_of_coins3), color='red', fontsize=12)
        axes[3, 4].set_title('circles from Adaptive Cannied', color='blue', fontsize=10)
        axes[3, 4].axis('off')


    # Plot the image with merged contours
        axes[4, 1].imshow(image_with_merged_contours, cmap='gray')
        axes[4, 1].text(10, 30, 'Total Coins: ' + str(number_of_coin), color='red', fontsize=12)
        axes[4, 1].set_title('Merged Contours', color='blue', fontsize=10)
        axes[4, 1].axis('off')


    
        plt.tight_layout()
        plt.show()

    elif display==False:
        return merged_contours, number_of_coin

def main():
    image_path = "dataset/images/0.jpg"
    #model_test(image_path, display=True)

    #image_path = "F:\\France\\paris_cite\\S2\\image\\projet\\CoinDetector\\dataset\\images\\10.jpg"
    model_test(image_path, display=True)
"""
    merged_contours, number_of_coin = model_test(image_path)
    print("Merged Contours:", merged_contours)
    print("Total Coins Detected:", number_of_coin)


"""



if __name__ == "__main__":
    main()
