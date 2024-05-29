from code.dataset import CoinDataset
from code.evaluation.evaluation import evaluate_dataset
from code.model.model import model_pipeline

def main():
    # Le chemin d'accès aux images et aux annotations
    image_dir = "dataset/images"
    annotation_dir = "dataset/labels"

    # Création de l'objet dataset
    dataset = CoinDataset(image_dir, annotation_dir)

    # Demander à l'utilisateur ce qu'il veut faire
    user_choice = input("Voulez-vous évaluer le modèle ou l'essayer sur une image ? (evaluer/essayer): ")

    if user_choice == 'evaluer':
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

    elif user_choice == 'essayer':
        # Demander à l'utilisateur quelle image essayer
        image_number = -1
        while image_number < 0 or image_number > 288:
            image_number = int(input("Entrez le numéro de l'image à essayer (0-288): "))

        # Demander à l'utilisateur s'il veut afficher les détails
        display_choice = input("Voulez-vous afficher les détails ? (oui/non): ")
        details = display_choice.lower() == 'oui'

        # Construire le chemin de l'image et exécuter le pipeline du modèle
        image_path = f"dataset/images/{image_number}.jpg"
        model_pipeline(image_path, display=True, details=details)

if __name__ == "__main__":
    main()