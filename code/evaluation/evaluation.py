import json
import cv2
import numpy as np

from code.dataset import load_annotations
from code.model.model import *


def calculate_iou(pred_box, gt_box):
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
  """
  Calculates F1 score, precision, and recall.

  Args:
    predictions: A list of predicted bounding boxes.
    ground_truths: A list of ground truth bounding boxes.
    threshold: IoU threshold for considering a prediction as a true positive.

  Returns:
    A tuple containing F1 score, precision, and recall.
  """

  positives = 0
  true_positives = 0
  false_negatives = 0
  matched_gt = set()  # Keep track of matched ground truth annotations

  for prediction in predictions:
    matched = False
    for ground_truth in ground_truths:
      if ground_truth not in matched_gt:  # Avoid counting same GT multiple times
        iou = calculate_iou(prediction, ground_truth)
        if iou >= threshold:
          positives += 1
          true_positives += 1
          matched_gt.add(ground_truth)
          matched = True
          break
    if not matched:
      positives += 1  # Count False Positives
      false_negatives += 1  # Add False Negatives


  # Calculate precision, recall, and F1 score (assuming you have a calculate_iou function)
  if positives == 0:
    return 0, 0, 0  # Handle division by zero
  precision = true_positives / positives
  recall = true_positives / len(ground_truths)
  f1_score = 2 * (precision * recall) / (precision + recall)

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
  Creates a confusion matrix for the evaluation.

  Args:
    predictions: A list of predicted bounding boxes.
    ground_truths: A list of ground truth bounding boxes.
    threshold: IoU threshold for considering a prediction as a true positive.

  Returns:
    A confusion matrix represented as a dictionary.
  """

  confusion_matrix = {
      "True Positives": 0,
      "False Positives": 0,
      "False Negatives": 0,
      "True Negatives": 0,  # Not applicable in object detection with binary classes
  }

  matched_gt = set()  # Initialize an empty set to track matched ground truths

  for prediction in predictions:
    matched = False
    for ground_truth in ground_truths:
      if ground_truth not in matched_gt:  # Avoid counting same GT multiple times
        iou = calculate_iou(prediction, ground_truth)
        if iou >= threshold:
          if ground_truth in matched_gt:
            confusion_matrix["False Positives"] += 1  # Predicted as coin, but not a real coin
          else:
            confusion_matrix["True Positives"] += 1  # Predicted as coin, and is a real coin
          matched = True
          matched_gt.add(ground_truth)  # Add matched ground truth to the set
          break
    if not matched and ground_truth not in matched_gt:
      confusion_matrix["False Negatives"] += 1  # Not predicted as coin, but is a real coin

  return confusion_matrix


def evaluate_image(image_path, annotation_path, threshold=0.5):
    predictions,_ = model_test(image_path)

    with open(annotation_path, 'r') as f:
        annotations = json.load(f)
        ground_truths = []
        for shape in annotations["shapes"]:
            x, y = shape["points"][0]
            radius = cv2.norm(np.array(shape["points"][0]) - np.array(shape["points"][1])) / 2
            ground_truths.append((x, y, radius))

    #print("Predictions:", predictions)
    #print("Ground Truths:", ground_truths)

    f1_score, precision, recall = calculate_f1_score(predictions, ground_truths, threshold)
    mde = calculate_mde(predictions, ground_truths)
    confusion_matrix = create_confusion_matrix(predictions, ground_truths, threshold)

    print("f1_score, precision, recall:", f1_score, precision, recall)
    print("Mean Detection Error (MDE):", mde)
    print("Nb Detected Coins:", len(predictions))
    print("Nb Annotated Coins:", len(ground_truths))
    print("Confusion Matrix:", confusion_matrix)

    return {"F1 Score": f1_score, "Mean Detection Error (MDE)": mde, "Nb Detected Coins": len(predictions), "Nb Annotated Coins": len(ground_truths), "confusion_matrix": confusion_matrix}



def evaluate_dataset(image_paths, annotation_paths, threshold=0.5):
    """
    Evaluates the model performance on a given dataset.

    Args:
        image_paths (list): List of paths to the images.
        annotation_paths (list): List of paths to the annotations.
        threshold (float, optional): Threshold for IoU to consider a detection correct (default: 0.5).

    Returns:
        dict: Dictionary containing average F1 score, MDE, and number of detected/annotated coins.
    """
    total_f1 = 0
    total_mde = 0
    total_detected = 0
    total_annotated = 0
    confusion_matrix = {"True Positives": 0, "False Positives": 0, "False Negatives": 0}  


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
    image_path = "dataset\\images\\40.jpg"
    annotation_path = "dataset\\labels\\40.json"

    evaluate_image(image_path, annotation_path)



if __name__ == "__main__":
    main()