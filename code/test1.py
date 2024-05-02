import json
import numpy as np
import os
import cv2
import glob
from sklearn.model_selection import train_test_split

class CoinDataset:
    def __init__(self, image_dir, annotation_dir, test_size=0.2, validation_size=0.2):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.test_size = test_size
        self.validation_size = validation_size

        self.image_paths = [os.path.join(image_dir, f) for f in glob.glob(os.path.join(image_dir, "*.jp*"))]
        self.annotation_paths = [
            os.path.join(annotation_dir, os.path.splitext(os.path.basename(p))[0] + ".json")
            for p in self.image_paths]

        print(f"Number of image paths: {len(self.image_paths)}")
        print(f"Number of annotation paths: {len(self.annotation_paths)}")

        self._split_data()

    def _split_data(self):
        image_annotation_pairs = [(path, annotation_path)
                                  for path, annotation_path in zip(self.image_paths, self.annotation_paths)
                                  if os.path.exists(annotation_path)]

        if len(image_annotation_pairs) < len(self.image_paths):
            print(f"Warning: Excluding {len(self.image_paths) - len(image_annotation_pairs)} images without annotations.")

        images, annotations = zip(*image_annotation_pairs)

        X_train, X_test, y_train, y_test = train_test_split(
            images, annotations, test_size=self.test_size, random_state=42
        )

        X_val, X_test, y_val, y_test = train_test_split(
            X_test, y_test, test_size=self.validation_size / (1 - self.test_size), random_state=42
        )

        self.train_images = X_train
        self.train_annotations = y_train
        self.val_images = X_val
        self.val_annotations = y_val
        self.test_images = X_test
        self.test_annotations = y_test

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of bounds")

        image_path = self.train_images[idx]
        annotation_path = self.train_annotations[idx]
        image = cv2.imread(image_path)

        with open(annotation_path, 'r') as f:
            annotation = json.load(f)

        return image, annotation

def load_annotations(annotation_path):
    with open(annotation_path, 'r') as f:
        annotations = json.load(f)
        ground_truths = []
        for shape in annotations["shapes"]:
            x, y = shape["points"][0]
            radius = cv2.norm(np.array(shape["points"][0]) - np.array(shape["points"][1])) / 2
            ground_truths.append((x, y, radius))
    return ground_truths

def main():
    image_path = "F:\\France\\paris_cite\\S2\\image\\projet\\CoinDetector\\dataset\\images\\40.jpg"
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
