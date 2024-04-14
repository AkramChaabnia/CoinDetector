import json
from sklearn.model_selection import train_test_split
import os
import cv2

from code.evaluation.evaluation import calculate_f1_score, calculate_mde
from code.model.model import model_test  # for image loading


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

        # Load image and annotation paths
        self.image_paths = self._load_paths(image_dir, ".jpg")
        self.annotation_paths = self._load_paths(annotation_dir, ".json")

        # Split data into training, validation, and test sets
        self._split_data()

    def _load_paths(self, directory, extension):
        """
        Loads paths to images or annotations from a directory.

        Args:
            directory (str): Path to the directory.
            extension (str): File extension to filter (e.g., ".jpg", ".json").

        Returns:
            list: List of file paths.
        """
        paths = []
        for filename in os.listdir(directory):
            if filename.endswith(extension):
                path = os.path.join(directory, filename)
                paths.append(path)
        return paths

    def _split_data(self):
        """
        Splits image and annotation paths into training, validation, and test sets.
        """
        X_train, X_test, y_train, y_test = train_test_split(
            self.image_paths, self.annotation_paths, test_size=self.test_size, random_state=42
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

        image_path = self.image_paths[idx]
        annotation_path = self.annotation_paths[idx]

        # Load image (replace with your image loading logic)
        image = cv2.imread(image_path)

        # Load annotation (replace with your JSON parsing logic)
        with open(annotation_path, 'r') as f:
            annotation = json.load(f)
        # ... extract ground truth information from annotation

        return image, annotation


def load_annotations(annotation_path):
    """
    Function to load annotation data from a JSON file.

    This function should be implemented based on the format of your annotations.

    Args:
        annotation_path (str): Path to the JSON annotation file.

    Returns:
        list: List of ground truth information (e.g., bounding boxes, keypoints).
    """
    # Implement logic to parse your specific JSON annotation format
    # ...
    pass


def evaluate_dataset(model_test, dataset, threshold=0.5):
    """
    Evaluates the model performance on a given dataset.

    Args:
        model_test (function): Function that takes an image path and returns predictions.
        dataset (CoinDataset): Instance of CoinDataset for the data to evaluate on.
        threshold (float, optional): Threshold for IoU to consider a detection correct (default: 0.5).

    Returns:
        dict: Dictionary containing average F1 score, MDE, and number of detected/annotated coins.
    """
    total_f1 = 0
    total_mde = 0
    total_detected = 0
    total_annotated = 0

    for image_path, label_path in zip(dataset.image_paths, dataset.annotation_paths):
        predictions, _ = model_test(image_path)  # Assuming model_test returns predictions and discards other outputs
        ground_truths = load_annotations(label_path)

        f1_score = calculate_f1_score(predictions, ground_truths, threshold)
        mde = calculate_mde(predictions, ground_truths)
        total_f1 += f1_score
        total_mde += mde
        total_detected += len(predictions)
        total_annotated += len(ground_truths)

    average_f1 = total_f1 / len(dataset)
    average_mde = total_mde / len(dataset)
    return {
        "Average F1 Score": average_f1,
        "Mean Detection Error (MDE)": average_mde,
        "Nb Detected Coins": total_detected,
        "Nb Annotated Coins": total_annotated,
    }


# Now you can use the CoinDataset class in your main script:

if __name__ == "__main__":
    # Example usage
    image_dir = "dataset/images"
    annotation_dir = "dataset/labels"
    dataset = CoinDataset(image_dir, annotation_dir)

    # Access training data
    train_images, train_annotations = dataset.get_train_data()

    # ... (Your training logic using train_images and train_annotations)

    # Evaluate on validation set
    val_f1_score, val_mde, etc = evaluate_dataset(model_test, dataset, split="val")
    print(f"Validation F1 Score: {val_f1_score}")
    print(f"Validation MDE: {val_mde}")
