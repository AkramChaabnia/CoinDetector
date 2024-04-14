import json
import cv2
import numpy as np

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
    for prediction in predictions:
        for ground_truth in ground_truths:
            iou = calculate_iou(prediction, ground_truth)
            if iou >= threshold:
                positives += 1
                if iou == 1.0:
                    true_positives += 1
                    break
    if positives == 0:
        return 0
    precision = true_positives / positives
    recall = true_positives / len(ground_truths)
    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score


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


def evaluate_image(image_path, annotation_path, threshold=0.5):
    predictions,_ = model_test(image_path)

    with open(annotation_path, 'r') as f:
        annotations = json.load(f)
        ground_truths = []
        for shape in annotations["shapes"]:
            x, y = shape["points"][0]
            radius = cv2.norm(np.array(shape["points"][0]) - np.array(shape["points"][1])) / 2
            ground_truths.append((x, y, radius))

    print("Predictions:", predictions)
    print("Ground Truths:", ground_truths)

    f1_score = calculate_f1_score(predictions, ground_truths, threshold)
    mde = calculate_mde(predictions, ground_truths)

    print("F1 Score:", f1_score)
    print("Mean Detection Error (MDE):", mde)
    print("Nb Detected Coins:", len(predictions))
    print("Nb Annotated Coins:", len(ground_truths))

    return {"F1 Score": f1_score, "Mean Detection Error (MDE)": mde, "Nb Detected Coins": len(predictions), "Nb Annotated Coins": len(ground_truths)}

def main():
    image_path = "dataset\\images\\40.jpg"
    annotation_path = "dataset\\labels\\40.json"

    evaluate_image(image_path, annotation_path)



if __name__ == "__main__":
    main()