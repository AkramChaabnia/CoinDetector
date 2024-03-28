import os
import random
import shutil

def preprocess_data(image_dir, json_dir, output_dir):
    # Créer les répertoires pour les ensembles d'entraînement, de validation et de test
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    test_dir = os.path.join(output_dir, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Appeler la fonction pour répartir les données
    train_files, val_files, test_files = split_dataset(json_dir)

    # Déplacer les fichiers vers les répertoires correspondants
    move_files(image_dir, json_dir, train_files, train_dir)
    move_files(image_dir, json_dir, val_files, val_dir)
    move_files(image_dir, json_dir, test_files, test_dir)

def move_files(image_dir, json_dir, files, output_dir):
    for file in files:
        image_path = os.path.join(image_dir, file.split('.')[0] + '.jpg')  # Changer l'extension si nécessaire
        json_path = os.path.join(json_dir, file)

        # Vérifier si le fichier JSON existe
        if not os.path.exists(json_path):
            continue

        # Copier l'image et le fichier JSON dans le répertoire de sortie
        shutil.copy(image_path, output_dir)
        shutil.copy(json_path, output_dir)

def split_dataset(data_path, shuffle=True):
    ratio = (60, 20, 20)
    # récuperer tous les fichiers Json
    dataset_files = [f for f in os.listdir(data_path) if f.endswith('.json')]
    if shuffle:
        random.shuffle(data_path)

    total_files = len(dataset_files)
    train_size = int(total_files * ratio[0] / 100)
    val_size = int(total_files * ratio[1] / 100)
    test_size = total_files - train_size - val_size

    train_files = dataset_files[:train_size]
    val_files = dataset_files[train_size:train_size + val_size]
    test_files = dataset_files[train_size + val_size:]

    return train_files, val_files, test_files
