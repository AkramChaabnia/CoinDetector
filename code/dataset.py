import json
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
import os
import cv2

class CoinDataset:
  """
    Cette classe représente un ensemble de données de pièces de monnaie.

    Attributs:
    image_dir (str): Le répertoire contenant les images.
    annotation_dir (str): Le répertoire contenant les fichiers d'annotations.
    test_size (float): La proportion des données à utiliser comme ensemble de test.
    image_paths (list): Une liste de chemins vers les images.
    annotation_paths (list): Une liste de chemins vers les fichiers d'annotations.
    train_images (list): Une liste de chemins vers les images d'entraînement.
    train_annotations (list): Une liste de chemins vers les annotations d'entraînement.
    val_images (list): Une liste de chemins vers les images de validation.
    val_annotations (list): Une liste de chemins vers les annotations de validation.
    test_images (list): Une liste de chemins vers les images de test.
    test_annotations (list): Une liste de chemins vers les annotations de test.

    Méthodes:
    __init__(self, image_dir, annotation_dir, test_size=0.4): Initialise l'objet CoinDataset.
    _split_data(self): Vérifie les paires image-annotation et divise les données en ensembles d'entraînement, de validation et de test.
    __len__(self): Retourne le nombre total d'images dans l'ensemble de données.
    __getitem__(self, idx): Charge et retourne l'image et l'annotation à l'indice spécifié.
  """
  def __init__(self, image_dir, annotation_dir, test_size=0.4):
    self.image_dir = image_dir
    self.annotation_dir = annotation_dir
    self.test_size = test_size

    image_dir_path = Path(image_dir)
    self.image_paths = [str(f) for f in image_dir_path.glob("*.jp*")]

    annotation_dir_path = Path(annotation_dir)
    self.annotation_paths = [str(annotation_dir_path / (Path(p).stem + ".json")) for p in self.image_paths]

    print(f"Number of image paths: {len(self.image_paths)}")
    print(f"Number of annotation paths: {len(self.annotation_paths)}")

    # divise les données en training, validation et test et exclut les images sans annotations
    self._split_data()

  def _split_data(self):
    print("Checking image-annotation pairs...")
    image_paths_without_annotations = []
    image_annotation_pairs = [(path, annotation_path)
                              for path, annotation_path in zip(self.image_paths, self.annotation_paths)
                              if os.path.exists(annotation_path)]
    for path in self.image_paths:
        if path not in [pair[0] for pair in image_annotation_pairs]:
            image_paths_without_annotations.append(path)

    print(f"Found {len(image_annotation_pairs)} image-annotation pairs.")
    if len(image_annotation_pairs) < len(self.image_paths):
        print(f"Warning: Excluding {len(self.image_paths) - len(image_annotation_pairs)} images without annotations.")
        print(f"Image paths without annotations: {image_paths_without_annotations}")
    
    images, annotations = zip(*image_annotation_pairs)
    # divise les données en 60% entrainement et 40% temporaire
    X_train, X_temp, y_train, y_temp = train_test_split(
        images, annotations, test_size=0.4, random_state=42
    )

    # divise les données temporaires en 50% validation et 50% test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )
    self.train_images = X_train
    self.train_annotations = y_train
    self.val_images = X_val
    self.val_annotations = y_val
    self.test_images = X_test
    self.test_annotations = y_test
    print(f"Training set size: {len(self.train_images)}")
    print(f"Validation set size: {len(self.val_images)}")
    print(f"Test set size: {len(self.test_images)}")

  def __len__(self):
        return len(self.image_paths)

  def __getitem__(self, idx):
    if idx < 0 or idx >= len(self):
        raise IndexError("Index out of bounds")
    
    # charger les images et les annotations
    image_path = self.train_images[idx]
    annotation_path = self.train_annotations[idx]

    print(f"Loading image: {image_path}")
    image = cv2.imread(image_path.replace("\\", "/"))
    print(f"Loading annotation: {annotation_path}")
    with open(annotation_path, 'r') as f:
        annotation = json.load(f)

    return image, annotation


def load_annotations(annotation_path):
    """
    Cette fonction charge les annotations à partir d'un fichier JSON.

    Paramètres:
    annotation_path (str): Le chemin vers le fichier d'annotations à charger.

    La fonction ouvre le fichier d'annotations, lit le contenu JSON et extrait les informations de forme pour chaque annotation. Les formes dans ce contexte sont des cercles. Pour chaque cercle, elle calcule le rayon comme la moitié de la norme de la différence entre les deux points du cercle, puis ajoute un tuple (x, y, rayon) à la liste des vérités terrain.

    Retourne:
    ground_truths (list): Une liste de tuples (x, y, rayon) représentant les vérités terrain pour chaque cercle dans le fichier d'annotations.
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
    image_dir = "dataset/images"
    annotation_dir = "dataset/labels"

    # creer un objet dataset
    dataset = CoinDataset(image_dir, annotation_dir)

    # acceder aux images et annotations
    image, annotation = dataset[10]
    print(f"Image shape: {image.shape}")


    # tester la fonction load_annotations
    annotation_path = dataset.annotation_paths[10]
    loaded_annotation = load_annotations(annotation_path)
    print(f"Loaded annotation (example): {loaded_annotation[0]}")


if __name__ == "__main__":
    main()
