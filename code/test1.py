import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import data, io, img_as_ubyte
from skimage.filters import threshold_multiotsu


def main():
    image_path = "dataset/reste/203.jpg"
    image = io.imread(image_path)

# Convert the image to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Now use the grayscale image with threshold_multiotsu
    thresholds = threshold_multiotsu(image, classes=2)    #print the thresholds
    #print(thresholds)

    # Using the threshold values, we generate the three regions.
    regions = np.digitize(image, bins=thresholds)
    #output = img_as_ubyte(regions)

    #print the unique values of the regions to see how the image has been divided and the color of that region in the segmented image

    print(np.unique(regions))   

    for i in np.unique(regions):
        region_pixels = np.where(regions == i)
        print(f"Pixels in region {i}: {region_pixels}")


    #Let us look at the input image, thresholds on thehistogram and final segmented image
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 3.5))

# Plotting the original image.
    ax[0].imshow(image, cmap='gray')
    ax[0].set_title('Original')
    ax[0].axis('off')

    # Plotting the histogram and the two thresholds obtained from
    # multi-Otsu.
    ax[1].hist(image.ravel(), bins=255, histtype='step', color='black')
    ax[1].set_title('Histogram')
    for thresh in thresholds:
        ax[1].axvline(thresh, color='r')

    # Plotting the Multi Otsu result.
    ax[2].imshow(regions, cmap='Accent')
    ax[2].set_title('Multi-Otsu result')
    ax[2].axis('off')


    plt.subplots_adjust()

    plt.show()

if __name__ == "__main__":
    main()
