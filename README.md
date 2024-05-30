<p align="center">
  <a href="" rel="noopener">
    <img src="" alt="Logo du projet">
  </a>
</p>
<h3 align="center">CoinDetector</h3>

<div align="center">

[![Status](https://img.shields.io/badge/status-actif-success.svg)]()
[![GitHub Issues](https://img.shields.io/github/issues/AkramChaabnia/CoinDetector.svg)](https://github.com/AkramChaabnia/CoinDetector/issues)
[![GitHub Pull Requests](https://img.shields.io/github/issues-pr/AkramChaabnia/CoinDetector.svg)](https://github.com/AkramChaabnia/CoinDetector/pulls)
[![University](https://img.shields.io/badge/University-Paris%20Cité%20Université-%23A6192E)](https://u-paris.fr)

</div>

---

<p align="center"> CoinDetector est un projet conçu pour détecter les pièces dans les images et évaluer les performances de l'algorithme de détection à l'aide de diverses métriques. Il a été développé dans le cadre d'un module d'image (IF06X070) à l'Université de Paris Cité pour l'annee universitaire 2023/2024.
  <br> 
</p>

## 📝 Table des Matières

- [À propos](#about)
- [Structure du Code](#code_structure)
- [Évaluation](#evaluation)
- [Pipeline](#pipeline)
- [Prérequis](#prerequisites)
- [Commencer](#getting_started)
- [Utilisation](#usage)
- [Construit avec](#built_using)
- [Auteurs](#authors)
- [Remerciements](#acknowledgement)
- [Contact](#contact)

## 🧐 À propos <a name = "about"></a>

### Détection de Pièces de Monnaie

#### Problématique

La détection automatique des pièces de monnaie dans les images est un défi en vision par ordinateur, en raison des variations d'éclairage, de couleur et de texture. Ce projet vise à développer un système de détection précis capable de segmenter et d'identifier les pièces de monnaie dans diverses conditions d'image, à partir des méthodes classiques en vision par ordinateur.

#### Objectif

L'objectif de ce projet est de creer un model qui va détecter les pièces dans les images, de visualiser les détections et d'évaluer les performances de la détection à l'aide de métriques et aussi faire la classification de ses pieces detectees. Le projet inclut des méthodes pour le prétraitement, la segmentation, la détection des contours et l'évaluation.

#### Solution

Nous proposons une solution basée sur une pipeline de traitement d'image qui inclut le prétraitement, la segmentation, l'extraction de caractéristiques et le post-traitement pour identifier et compter les pièces de monnaie dans une image. Nous avons également commencé à explorer la classification en utilisant la méthode Gabor–Granger.

##### Méthodes Classiques Utilisées

- Flou Gaussien
- Correction Gamma
- Conversion en Niveaux de Gris
- Flou Médian
- Égalisation de l'Histogramme Adaptatif
- Seuillage Otsu
- Seuillage Multi-Otsu
- Segmentation Basée sur les Couleurs
- Seuillage Adaptatif
- Détection des Contours avec Canny
- Détection de Cercles avec Hough Transform
- Gabor–Granger
- LBP

## 🚀 Structure du Code <a name = "code_structure"></a>

Le projet se compose des fichiers et répertoires suivants :

- `dataset/`
  - `images/`: Répertoire contenant les images de pièces.
  - `labels/`: Répertoire contenant les annotations pour les images de pièces.
- `code/`
  - `model/`
    - `__init__.py`
    - `preprocess.py`: Contient les fonctions de prétraitement des images.
    - `classification.py`: Contient les fonctions de classification des images.
    - `postprocessing.py`: Contient les fonctions de post-traitement des caractéristiques détectées.
    - `segmentation.py`: Contient les fonctions de segmentation des images.
    - `feature_extraction.py`: Contient les fonctions d'extraction des caractéristiques.
    - `feature_detection.py`: Contient les fonctions de détection des caractéristiques.
    - `model.py`: Contient la fonction principale du pipeline du modèle.
  - `evaluation/`
    - `__init__.py`
    - `evaluation.py`: Contient les fonctions d'évaluation du modèle.
  - `dataset.py`: Contient les fonctions de chargement du jeu de données.
  - `main.py`: Contient la fonction principale pour exécuter le projet.
  - `utils.py`: Contient des fonctions utilitaires.

## 🎈 Évaluation <a name="evaluation"></a>

Le projet utilise les métriques suivantes pour évaluer les performances de l'algorithme de détection de pièces :

- **F1 Score**: Mesure la précision des pièces détectées.
- **Erreur Moyenne de Détection (MDE)**: Mesure la distance moyenne entre les centres des pièces détectées et la vérité terrain.
- **Matrice de Confusion**: Fournit le compte des vrais positifs, faux positifs et faux négatifs.

Pour être capable d'utiliser ces métriques, nous utilisons également la méthode Intersection sur Union (IoU). L'IoU est une mesure de l'overlap entre deux zones. Dans notre cas, il s'agit de l'overlap entre la zone de détection de la pièce et la vérité terrain. L'IoU est utilisé pour déterminer si une détection est un vrai positif ou un faux positif. En général, si l'IoU d'une détection avec la vérité terrain est supérieure à un certain seuil (souvent 0.5), la détection est considérée comme un vrai positif. L'utilisation de l'IoU permet d'avoir une mesure plus robuste de la performance du modèle, car elle prend en compte à la fois la localisation et la taille des détections.

### Résultats de l'Évaluation

#### Ensemble de Validation

- **F1 Score Moyen**: 0.801
- **Erreur Moyenne de Détection (MDE)**: 62.66
- **Nombre de Pièces Détectées**: 259
- **Nombre de Pièces Annotées**: 311
- **Matrice de Confusion**:
  - Vrais Positifs: 227
  - Faux Positifs: 32
  - Faux Négatifs: 84
  - Vrais Négatifs: 0

#### Ensemble d'Entraînement

- **F1 Score Moyen**: 0.829
- **Erreur Moyenne de Détection (MDE)**: 75.33
- **Nombre de Pièces Détectées**: 958
- **Matrice de Confusion**:
  - Vrais Positifs: 778
  - Faux Positifs: 180
  - Faux Négatifs: 202
  - Vrais Négatifs: 0

#### Ensemble de Test

- **F1 Score Moyen**: 0.829
- **Erreur Moyenne de Détection (MDE)**: 49.70
- **Nombre de Pièces Détectées**: 341
- **Nombre de Pièces Annotées**: 358
- **Matrice de Confusion**:
  - Vrais Positifs: 278
  - Faux Positifs: 63
  - Faux Négatifs: 80
  - Vrais Négatifs: 0

Ces résultats montrent que le modèle a de bonnes performances globales en termes de précision (F1 Score) et d'erreur moyenne de détection (MDE) sur les ensembles de validation, d'entraînement et de test. Cependant, il reste des faux positifs et des faux négatifs qui pourraient être réduits pour améliorer encore les performances du modèle.

## 🛠️ Pipeline <a name = "pipeline"></a>

Le pipeline suit ces étapes :

1. **Prétraitement**
   - Application de Flou Gaussien : Réduction du bruit
   - Correction Gamma : Ajustement de l'éclairage
   - Conversion en Niveaux de Gris : Simplification de l'image
   - Flou Médian : Réduction du bruit
   - Égalisation de l'Histogramme Adaptatif : Amélioration du contraste
2. **Segmentation**
   - Seuillage Otsu
   - Seuillage Multi-Otsu
   - Segmentation Basée sur les Couleurs
   - Seuillage Adaptatif
3. **Détection de Contours avec Canny**
   - Application de Canny sur chaque image segmentée
4. **Filtrage des Contours**
   - Conservation des contours circulaires uniquement
5. **Fusion des Résultats**
   - Fusion des cercles détectés à partir des différentes méthodes de segmentation
6. **Visualisation des Résultats**
   - Affichage des résultats avec des exemples concrets

Vous pouvez tester le pipeline avec :

```bash
python -m code.model.model
```

## 🛠️ Prérequis <a name = "prerequisites"></a>

Assurez-vous d'avoir les éléments suivants installés :

- Python 3.7 ou supérieur
- OpenCV
- NumPy
- Matplotlib

## 🏁 Commencer <a name = "getting_started"></a>

Pour démarrer avec le projet, suivez ces étapes :

1. Clonez le dépôt :
   ```bash
   git clone https://github.com/AkramChaabnia/CoinDetector.git
   cd CoinDetector
   ```
2. Installez les packages requis :
   ```bash
   pip install -r requirements.txt
   ```

## 🎈 Utilisation <a name="usage"></a>

Pour exécuter le script principal :

```bash
python -m code.main
```

Vous serez invité à choisir entre évaluer le modèle ou le tester sur une seule image.

- **Évaluer le Modèle**: Sélectionnez un ensemble de données (train, val, test) à évaluer.
- **Tester sur une Seule Image**: Fournissez le numéro de l'image et choisissez entre afficher l'image résultat avec les pièces de monnaie détectées encerclées (exemple 1) ou afficher les étapes de traitement effectuées par notre modèle, de l'image originale au résultat final (exemple 2).

### Exemple 1: Résultat de la détection des pièces de monnaie

<p align="center">
  <img src="Figure_1.png" alt="Résultat de la détection">
</p>

### Exemple 2: Application du modèle avec les étapes de traitement

<p align="center">
  <img src="Figure_2.png" alt="Résultat de la détection">
</p>

## ⛏️ Construit avec <a name = "built_using"></a>

- Python - Langage de programmation
- OpenCV - Bibliothèque de vision par ordinateur
- NumPy - Bibliothèque de calcul numérique
- Matplotlib - Bibliothèque de visualisation

## ✍️ Auteurs <a name = "authors"></a>

- Akram Chaabnia - Travail initial
- Mohamed Razi - Commencer l'implementation de la classification

## 🎉 Remerciements <a name = "acknowledgement"></a>

Nous tenons à exprimer notre profonde gratitude à notre professeur, M. Sylvain Lobry, qui est le responsable de ce projet. Son expertise et ses conseils ont été inestimables tout au long de ce travail. Ce projet a été inspiré par le module d'image (IF06X070) à l'Université de Paris Cité.

## 📞 Contact <a name = "contact"></a>

Si vous avez des questions, des suggestions ou souhaitez simplement entrer en contact, voici comment me joindre :

- Email : akram.chaabnia25@gmail.com
- LinkedIn : [akram-chaabnia](https://www.linkedin.com/in/akram-chaabnia/)
- GitHub : [AkramChaabnia](https://github.com/AkramChaabnia)

N'hésitez pas à me contacter !
