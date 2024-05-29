from code.dataset import CoinDataset
from code.evaluation.evaluation import evaluate_dataset

def main():
    # Le chemin d'accès aux images et aux annotations
    image_dir = "dataset/images"
    annotation_dir = "dataset/labels"

    # Création de l'objet dataset
    dataset = CoinDataset(image_dir, annotation_dir)

    # Demander à l'utilisateur quel set évaluer
    set_choice = ""
    while set_choice not in ['train', 'val', 'test']:
        set_choice = input("Choisissez le set à évaluer (train, val, test): ")

    # Charger les images et les annotations en fonction du choix de l'utilisateur
    if set_choice == 'train':
        images = dataset.train_images
        annotations = dataset.train_annotations
    elif set_choice == 'val':
        images = dataset.val_images
        annotations = dataset.val_annotations
    else:
        images = dataset.test_images
        annotations = dataset.test_annotations

    # Évaluation des images
    evaluation_results = evaluate_dataset(images, annotations)

    # Afficher les résultats de l'évaluation
    print(f"{set_choice.capitalize()} Set Evaluation:")
    for metric, value in evaluation_results.items():
        print(f"\t{metric}: {value}")

if __name__ == "__main__":
    main()