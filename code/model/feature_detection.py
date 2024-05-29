import os
import cv2
from code.model.feature_extraction import extract_gabor_features, extract_lbp_features




def extract_texture_features(image_path):
    # Charger l'image
    image = cv2.imread(image_path)

    # Extraire les caractéristiques de Gabor
    gabor_features = extract_gabor_features(image)

    # Extraire les caractéristiques LBP
    lbp_features = extract_lbp_features(image)

    return gabor_features, lbp_features

def process_images_in_folder(folder_path):
    # Créer un dossier pour sauvegarder les résultats
    result_folder = os.path.join(folder_path, "texture_features")
    os.makedirs(result_folder, exist_ok=True)

    # Parcourir les fichiers dans le dossier
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # Construire le chemin complet de l'image
            image_path = os.path.join(folder_path, filename)

            # Extraire les caractéristiques de texture
            gabor_features, lbp_features = extract_texture_features(image_path)

            # Sauvegarder les résultats
            result_filename = os.path.splitext(filename)[0] + "_texture_features.txt"
            result_filepath = os.path.join(result_folder, result_filename)
            with open(result_filepath, "w") as file:
                file.write(f"Gabor features: {gabor_features}\n")
                file.write(f"LBP features: {lbp_features}\n")

            print(f"Features saved for {filename}")

def main():
    folder_path = "dataset/roi"
    process_images_in_folder(folder_path)

if __name__ == "__main__":
    main()
