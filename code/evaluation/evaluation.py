import cv2
import numpy as np
from code.dataset import load_annotations
from code.model.model import model_pipeline


def calculate_iou(pred_box, gt_box):
    """
    Cette fonction calcule l'Intersection sur Union (IoU) de deux cercles. 

    Paramètres:
    pred_box (tuple): Un triplet (x, y, r) représentant la zone prédite, où (x, y) est le centre du cercle et r est le rayon.
    gt_box (tuple): Un triplet (x, y, r) représentant la zone de vérité terrain.

    L'IoU est une mesure de l'overlap entre deux zones. 
    Ici, elle est utilisée pour comparer pred_box avec gt_box.
    La fonction retourne un score entre 0 et 1, où 1 signifie que les deux cercles se superposent parfaitement et 0 signifie qu'ils ne se superposent pas du tout.
    Nous allons utiliser ce score pour calculer le F1-score et la matrice de confusion, car ces métriques ne fonctionnent pas directement sur les résultats binaires.

    Retourne:
    iou (float): Le score IoU entre pred_box et gt_box. Il est compris entre 0 (pas de superposition) et 1 (superposition parfaite).
    """
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
    Cette fonction calcule le F1-score, la précision et le rappel pour une liste de prédictions et de vérités terrain.

    Paramètres:
    predictions (list): Une liste de zones prédites, chaque zone étant un triplet (x, y, r).
    ground_truths (list): Une liste de zones de vérité terrain, chaque zone étant un triplet (x, y, r).
    threshold (float): Le seuil d'IoU pour considérer une prédiction comme correcte. Par défaut, il est fixé à 0.5.

    Le F1-score est ensuite calculé comme la moyenne harmonique de la précision et du rappel.

    Retourne:
    f1_score (float): Le F1-score pour les prédictions, compris entre 0 (mauvais) et 1 (bon).
    precision (float): La précision des prédictions, c'est-à-dire le ratio des vrais positifs sur le total des positifs.
    recall (float): Le rappel des prédictions, c'est-à-dire le ratio des vrais positifs sur le total des vérités terrain.
    """
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


def create_confusion_matrix(predictions, ground_truths, threshold):
  """
    Cette fonction crée une matrice de confusion pour une liste de prédictions et de vérités terrain.

    Paramètres:
    predictions (list): Une liste de zones prédites, chaque zone étant un triplet (x, y, r).
    ground_truths (list): Une liste de zones de vérité terrain, chaque zone étant un triplet (x, y, r).
    threshold (float): Le seuil d'IoU pour considérer une prédiction comme correcte.

    La fonction parcourt chaque prédiction et chaque vérité terrain, et calcule l'IoU entre elles. Si l'IoU est supérieur au seuil, la prédiction est considérée comme un vrai positif. Si aucune vérité terrain ne correspond à la prédiction, elle est considérée comme un faux positif. Les vérités terrain qui ne correspondent à aucune prédiction sont considérées comme des faux négatifs.

    Retourne:
    confusion_matrix (dict): Un dictionnaire contenant le nombre de vrais positifs, de faux positifs et de faux négatifs.
  """

  confusion_matrix = {
      "True Positives": 0,
      "False Positives": 0,
      "False Negatives": 0,
  }

  matched_gt = set()

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


def calculate_mde(predictions, ground_truths):
    """
    Cette fonction calcule la Mean Distance Error (MDE) pour une liste de prédictions et de vérités terrain.

    Paramètres:
    predictions (list): Une liste de zones prédites, chaque zone étant un triplet (x, y, r).
    ground_truths (list): Une liste de zones de vérité terrain, chaque zone étant un triplet (x, y, r).

    La fonction parcourt chaque prédiction et chaque vérité terrain, et calcule la distance euclidienne entre elles. Pour chaque prédiction, elle trouve la vérité terrain la plus proche (c'est-à-dire avec la distance minimale). La MDE est ensuite la moyenne de ces distances minimales.

    Retourne:
    mde (float): La Mean Distance Error pour les prédictions. Elle est toujours positive, et une valeur plus petite indique une meilleure précision.
    """
    total_distance = 0
    for prediction in predictions:
        min_distance = float('inf')
        for ground_truth in ground_truths:
            distance = cv2.norm(np.array(prediction[0:2]) - np.array(ground_truth[0:2]))
            if distance < min_distance:
                min_distance = distance
        total_distance += min_distance
    return total_distance / len(predictions)


def evaluate_image(image_path, annotation_path, threshold=0.5, print=False):
    """
    Cette fonction évalue les prédictions de notre modèle sur une image en utilisant plusieurs métriques.

    Paramètres:
    image_path (str): Le chemin vers l'image à évaluer.
    annotation_path (str): Le chemin vers le fichier d'annotations de vérité terrain pour l'image.
    threshold (float): Le seuil d'IoU pour considérer une prédiction comme correcte. Par défaut, il est fixé à 0.5.
    print (bool): Si True, imprime les résultats de l'évaluation. Par défaut, il est fixé à False.

    La fonction charge l'image et les annotations, fait des prédictions sur l'image, puis calcule le F1-score, la matrice de confusion et la Mean Distance Error (MDE) pour les prédictions par rapport aux vérités terrain.

    Retourne:
    results (dict): Un dictionnaire contenant le F1-score, la MDE, le nombre de pièces détectées, le nombre de pièces annotées et la matrice de confusion.
    """

    predictions, _ = model_pipeline(image_path)
    ground_truths = load_annotations(annotation_path)

    f1_score, precision, recall = calculate_f1_score(predictions, ground_truths, threshold)
    confusion_matrix = create_confusion_matrix(predictions, ground_truths, threshold)
    mde = calculate_mde(predictions, ground_truths)

    if(print):
        print("f1_score, precision, recall:", f1_score, precision, recall)
        print("Mean Detection Error (MDE):", mde)
        print("Nb Detected Coins:", len(predictions))
        print("Nb Annotated Coins:", len(ground_truths))
        print("Confusion Matrix:", confusion_matrix)

    return {
        "F1 Score": f1_score,
        "Mean Detection Error (MDE)": mde,
        "Nb Detected Coins": len(predictions),
        "Nb Annotated Coins": len(ground_truths),
        "confusion_matrix": confusion_matrix,
    }


def evaluate_dataset(image_paths, annotation_paths, threshold=0.5):
    """
    Cette fonction évalue les prédictions de notre modèle sur un ensemble de données en utilisant plusieurs métriques.

    Paramètres:
    image_paths (list): Une liste de chemins vers les images à évaluer.
    annotation_paths (list): Une liste de chemins vers les fichiers d'annotations de vérité terrain pour les images.
    threshold (float): Le seuil d'IoU pour considérer une prédiction comme correcte. Par défaut, il est fixé à 0.5.

    La fonction parcourt chaque image et son fichier d'annotations correspondant, fait des prédictions sur l'image, puis calcule le F1-score, la matrice de confusion et la Mean Distance Error (MDE) pour les prédictions par rapport aux vérités terrain. Les résultats sont ensuite agrégés pour l'ensemble de données entier.

    Retourne:
    results (dict): Un dictionnaire contenant le F1-score moyen, la MDE moyenne, le nombre total de pièces détectées, le nombre total de pièces annotées et la matrice de confusion agrégée.
    """
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
    image_path="dataset/images/10.jpg"
    annotation_path="dataset/labels/10.json"
    evaluate_image(image_path, annotation_path)
"""
    image_paths = ["dataset/images/40.jpg", "dataset/images/41.jpg"]
    annotation_paths = ["dataset/labels/40.json", "dataset/labels/41.json"]
    evaluate_dataset(image_paths, annotation_paths)
"""


if __name__ == "__main__":
    main()
