import math
import cv2
from matplotlib import pyplot as plt
import numpy as np

from code.dataset import load_annotations
from code.model.model import model_test


def calculate_iou(pred_box, gt_box):
    #function to calculate the intersection over union of two circles
    x1, y1, r1 = pred_box
    x2, y2, r2 = gt_box
    d = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    if d > r1 + r2:
        return 0
    elif d <= abs(r1 - r2):
        return 1
    else:
        part1 = r1 ** 2 * np.arccos((d ** 2 + r1 ** 2 - r2 ** 2) / (2 * d * r1))
        part2 = r2 ** 2 * np.arccos((d ** 2 + r2 ** 2 - r1 ** 2) / (2 * d * r2))
        part3 = 0.5 * np.sqrt((-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2))
        iou = (part1 + part2 - part3) / (np.pi * min(r1, r2) ** 2)
        return iou


def calculate_f1_score(predictions, ground_truths, threshold=0.5):
    positives = 0
    true_positives = 0
    false_negatives = 0
    matched_gt = set()

    for prediction in predictions:
        matched = False
        for ground_truth in ground_truths:
            if ground_truth not in matched_gt:
                iou = calculate_iou(prediction, ground_truth)
                if iou >= threshold:
                    positives += 1
                    true_positives += 1
                    matched_gt.add(ground_truth)
                    matched = True
                    break
        if not matched:
            positives += 1
            false_negatives += 1

    if positives == 0:
        return 0, 0, 0
    precision = true_positives / positives
    recall = true_positives / len(ground_truths)
    if precision + recall != 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0

    return f1_score, precision, recall

def calculate_mde(predictions, ground_truths):
    total_distance = 0
    for prediction in predictions:
        min_distance = float('inf')
        for ground_truth in ground_truths:
            distance = cv2.norm(np.array(prediction[0:2]) - np.array(ground_truth[0:2]))
            if distance < min_distance:
                min_distance = distance
        total_distance += min_distance
    return total_distance / len(predictions)

def create_confusion_matrix(predictions, ground_truths, threshold):
  """
  Creates a confusion matrix for object detection evaluation.

  Args:
      predictions (list): List of predicted bounding boxes (center_x, center_y, radius).
      ground_truths (list): List of ground truth bounding boxes (center_x, center_y, radius).
      threshold (float): IoU threshold for considering a prediction as a true positive.

  Returns:
      dict: Confusion matrix dictionary with counts for True Positives, False Positives, and False Negatives.
  """

  confusion_matrix = {
      "True Positives": 0,
      "False Positives": 0,
      "False Negatives": 0,
  }

  matched_gt = set()  # Track matched ground truth annotations

  for prediction in predictions:
    matched = False
    for ground_truth in ground_truths:
      if ground_truth not in matched_gt:
        iou = calculate_iou(prediction, ground_truth)
        if iou >= threshold:
          confusion_matrix["True Positives"] += 1
          matched_gt.add(ground_truth)
          matched = True
          break
    if not matched:
      confusion_matrix["False Positives"] += 1

  confusion_matrix["False Negatives"] = len(ground_truths) - len(matched_gt)

  return confusion_matrix

def visualize_predictions(image, predictions, ground_truths, threshold):
    # Define colors
    green = (0, 255, 0)  # Green for True Positives
    red = (0, 0, 255)  # Red for False Positives
    blue = (255, 0, 0)  # Blue for Missed Detections (False Negatives)

    # Draw bounding boxes
    for prediction in predictions:
        x, y, r = prediction
        cv2.circle(image, (int(x), int(y)), int(r), green, thickness=2)

    for ground_truth in ground_truths:
        found = False
        for prediction in predictions:
            if calculate_iou(prediction, ground_truth) >= threshold:
                found = True
                break
        if not found:
            x, y, r = ground_truth
            cv2.circle(image, (int(x), int(y)), int(r), blue, thickness=2)

    return image

def evaluate_image(image_path, annotation_path, threshold=0.5):
    predictions, _ = model_test(image_path)
    ground_truths = load_annotations(annotation_path)
    image = cv2.imread(image_path)

    # Visualize predictions
    image_with_predictions = visualize_predictions(image.copy(), predictions, ground_truths, threshold)

    # Calculate evaluation metrics
    f1_score, precision, recall = calculate_f1_score(predictions, ground_truths, threshold)
    mde = calculate_mde(predictions, ground_truths)
    confusion_matrix = create_confusion_matrix(predictions, ground_truths, threshold)

    # Display evaluation metrics
    print("f1_score, precision, recall:", f1_score, precision, recall)
    print("Mean Detection Error (MDE):", mde)
    print("Nb Detected Coins:", len(predictions))
    print("Nb Annotated Coins:", len(ground_truths))
    print("Confusion Matrix:", confusion_matrix)

# Convert the BGR image to RGB
    image_with_predictions_rgb = cv2.cvtColor(image_with_predictions, cv2.COLOR_BGR2RGB)

# Display the image
    plt.figure(figsize=(10, 10))  # You can adjust the figure size to your preference
    plt.imshow(image_with_predictions_rgb)
    plt.axis('off')  # To hide the axis values
    plt.show()

    return {
        "F1 Score": f1_score,
        "Mean Detection Error (MDE)": mde,
        "Nb Detected Coins": len(predictions),
        "Nb Annotated Coins": len(ground_truths),
        "confusion_matrix": confusion_matrix,
        "visualized_image": image_with_predictions.copy()  # Return a copy to avoid modification
    }


def evaluate_dataset(image_paths, annotation_paths, threshold=0.5):
    total_f1 = 0
    total_mde = 0
    total_detected = 0
    total_annotated = 0
    confusion_matrix = {
        "True Positives": 0,
        "False Positives": 0,
        "False Negatives": 0,
        "True Negatives": 0,
    }

    for image_path, annotation_path in zip(image_paths, annotation_paths):
        result = evaluate_image(image_path, annotation_path, threshold)
        total_f1 += result["F1 Score"]
        total_mde += result["Mean Detection Error (MDE)"]
        total_detected += result["Nb Detected Coins"]
        total_annotated += result["Nb Annotated Coins"]
        confusion_matrix["True Positives"] += result["confusion_matrix"]["True Positives"]
        confusion_matrix["False Positives"] += result["confusion_matrix"]["False Positives"]
        confusion_matrix["False Negatives"] += result["confusion_matrix"]["False Negatives"]

    average_f1 = total_f1 / len(image_paths)
    average_mde = total_mde / len(image_paths)

    return {
        "Average F1 Score": average_f1,
        "Mean Detection Error (MDE)": average_mde,
        "Nb Detected Coins": total_detected,
        "Nb Annotated Coins": total_annotated,
        "Confusion Matrix": confusion_matrix
    }


def main():
    image_path="dataset/images/0.jpg"
    annotation_path="dataset/labels/0.json"
    evaluate_image(image_path, annotation_path)
"""
    image_paths = ["dataset/images/40.jpg", "dataset/images/41.jpg"]
    annotation_paths = ["dataset/labels/40.json", "dataset/labels/41.json"]
    evaluate_dataset(image_paths, annotation_paths)

"""


if __name__ == "__main__":
    main()
