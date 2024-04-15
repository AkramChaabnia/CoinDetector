import math
import cv2
import numpy as np

from code.dataset import load_annotations
from code.model.model import model_test


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


def calculate_distance(prediction, ground_truth):
    x1, y1, _ = prediction
    x2, y2, _ = ground_truth
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def create_confusion_matrix(predictions, ground_truths, threshold):
    confusion_matrix = {
        "True Positives": 0,
        "False Positives": 0,
        "False Negatives": 0,
        "True Negatives": 0,
    }

    for prediction in predictions:
        matched = False
        for ground_truth in ground_truths:
            iou = calculate_iou(prediction, ground_truth)
            if iou >= threshold:
                confusion_matrix["True Positives"] += 1
                matched = True
                break
        if not matched:
            confusion_matrix["False Positives"] += 1

    confusion_matrix["False Negatives"] = len(ground_truths) - confusion_matrix["True Positives"]

    return confusion_matrix

def evaluate_image(image_path, annotation_path, threshold=0.5):
    predictions,_ = model_test(image_path)
    ground_truths = load_annotations(annotation_path)

    print("Predictions:", predictions)
    print("Ground Truths:", ground_truths)

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
    image_path="dataset/images/40.jpg"
    annotation_path="dataset/labels/40.json"
    evaluate_image(image_path, annotation_path)
"""
    image_paths = ["dataset/images/40.jpg", "dataset/images/41.jpg"]
    annotation_paths = ["dataset/labels/40.json", "dataset/labels/41.json"]
    evaluate_dataset(image_paths, annotation_paths)

"""


if __name__ == "__main__":
    main()
