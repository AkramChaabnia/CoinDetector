import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_multiotsu
from scipy.signal import find_peaks
from model.preprocess import *

def main():
    # Lire l'image
    image_path = "dataset/reste/203.jpg"
    original_image = cv2.imread(image_path)

    # Preprocess the image
    processed_image = apply_gaussian_blur(original_image.copy())
    gamma_processed_image = apply_gamma_correction(processed_image, gamma=1.5)
    grayscale_image = convert_to_grayscale(gamma_processed_image.copy())
    hist_processed_image = apply_adaptive_histogram_equalization(grayscale_image.copy())

    # Calculer l'histogramme
    #hist = cv2.calcHist([hist_processed_image], [0], None, [256], [0, 256])
    hist = np.histogram(hist_processed_image.ravel(), bins=256)[0]



    # Détecter les pics significatifs dans l'histogramme
    peaks, _ = find_peaks(hist, height=4000, width=3, distance=40)

    print("Pics Détectés:", peaks)

    # Déterminer le nombre de classes pour la segmentation
    num_classes = len(peaks) + 1  # Ajouter 1 car le nombre de classes est le nombre de seuils + 1
    print("Nombre de Classes pour la Segmentation:", num_classes)

    # Appliquer la segmentation avec multi-Otsu
    thresholds = threshold_multiotsu(hist_processed_image, classes=num_classes)
    print("Seuils de Segmentation:", thresholds)

    # Appliquer la segmentation
    segmented_image = apply_segmentation(hist_processed_image, thresholds)


    # Afficher l'histogramme avec les pics détectés
    plt.subplot(1, 2, 1)
    plt.plot(hist)
    plt.title('Histogramme avec les Pics')
    for peak in peaks:
        plt.axvline(x=peak, color='r', linestyle='--')


    # Afficher l'image originale et l'image segmentée
    plt.subplot(1, 2, 2)
    plt.imshow(segmented_image, cmap='gray')
    plt.title('Image Segmentée')
    plt.axis('off')

    plt.show()


def apply_segmentation(hist_processed_image, thresholds):
    # Appliquer la segmentation basée sur les seuils
    #_, segmented_image = cv2.threshold(hist_processed_image, thresholds[0], 255, cv2.THRESH_BINARY)
    segmented_image = np.digitize(hist_processed_image, bins=thresholds)

    return segmented_image

if __name__ == "__main__":
    main()
