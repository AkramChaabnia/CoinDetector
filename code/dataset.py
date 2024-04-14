import json
import numpy as np
from sklearn.model_selection import train_test_split
import os
import cv2
import glob  # Using glob for case-insensitive matching

class CoinDataset:
  """
  Class to load and manage the coin detection dataset.
  """

  def __init__(self, image_dir, annotation_dir, test_size=0.2, validation_size=0.2):
    """
    Args:
      image_dir (str): Path to the directory containing images.
      annotation_dir (str): Path to the directory containing annotation files (JSON format).
      test_size (float, optional): Proportion of data for the test set (default: 0.2).
      validation_size (float, optional): Proportion of data for the validation set (default: 0.2).
    """
    self.image_dir = image_dir
    self.annotation_dir = annotation_dir
    self.test_size = test_size
    self.validation_size = validation_size

    # Load image and annotation paths (case-insensitive matching)
    self.image_paths = [os.path.join(image_dir, f) for f in glob.glob(os.path.join(image_dir, "*.jp*"))]
    self.annotation_paths = [
      os.path.join(annotation_dir, os.path.splitext(os.path.basename(p))[0] + ".json")
      for p in self.image_paths
    ]

    # Print path lengths for verification
    print(f"Number of image paths: {len(self.image_paths)}")
    print(f"Number of annotation paths: {len(self.annotation_paths)}")

    # Split data (exclude images without annotations)
    self._split_data()

  def _split_data(self):
    """
    Splits image and annotation paths into training, validation, and test sets.
    Excludes images without corresponding annotations.
    """
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
        """
        Returns the total number of images in the dataset (all splits combined).
        """
        return len(self.image_paths)

def __getitem__(self, idx):
    """
    Retrieves an image and its corresponding annotation based on the index.

    Args:
        idx (int): Index of the image-annotation pair to access.

    Returns:
        tuple: (image, annotation)
    """
    if idx < 0 or idx >= len(self):
        raise IndexError("Index out of bounds")

    image_path = self.train_images[idx]  # Assuming you want to access from train set (modify if needed)
    annotation_path = self.train_annotations[idx]  # Assuming you want to access from train set (modify if needed)

    # Load image (replace with your image loading logic)
    image = cv2.imread(image_path)

    # Load annotation (replace with your JSON parsing logic)
    with open(annotation_path, 'r') as f:
        annotation = json.load(f)
    # ... extract ground truth information from annotation

    return image, annotation


def load_annotations(annotation_path):
    """
    Function to load annotation data from a JSON file (assuming circle annotations).

    Args:
        annotation_path (str): Path to the JSON annotation file.

    Returns:
        list: List of ground truth information (circles represented as (x, y, radius) tuples).
    """
    with open(annotation_path, 'r') as f:
        annotations = json.load(f)
        ground_truths = []
        for shape in annotations["shapes"]:
            x, y = shape["points"][0]
            radius = cv2.norm(np.array(shape["points"][0]) - np.array(shape["points"][1])) / 2
            ground_truths.append((x, y, radius))
    return ground_truths




def main():
    # Define paths to your image and annotation directories
    image_dir = "dataset/images"
    annotation_dir = "dataset/labels"

    # Create a CoinDataset instance
    dataset = CoinDataset(image_dir, annotation_dir)


    # Access data using indexing (assuming training set)
    image, annotation = dataset[10]
    print(f"Image shape: {image.shape}")


    # Print some information from the annotation (assuming it's a list of dictionaries)
    for obj in annotation:
        print(f"Object class: {obj['class']}")
        print(f"Bounding box: {obj['bbox']}")  # Assuming bounding box format in your JSON

    # Test loading annotations using the separate function
    annotation_path = dataset.annotation_paths[20]
    loaded_annotation = load_annotations(annotation_path)
    print(f"Loaded annotation (example): {loaded_annotation[0]}")


if __name__ == "__main__":
    main()
