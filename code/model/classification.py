import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from code.model.model import model_pipeline



# Fonction pour extraire les régions d'intérêt
def extract_ROI(image_path):
    circles, _ = model_pipeline(image_path)
    image = cv2.imread(image_path)
    ROIs = []
    for circle in circles:
        x, y, radius = circle
        x1 = max(int(x - radius), 0)
        y1 = max(int(y - radius), 0)
        x2 = min(int(x + radius), image.shape[1])
        y2 = min(int(y + radius), image.shape[0])
        ROI = image[y1:y2, x1:x2]
        ROIs.append(ROI)
    return ROIs, image, circles

def extract_texture_features(roi):
    # Extraire les caractéristiques de Gabor
    gabor_features = extract_gabor_features(roi)
    
    # Extraire les caractéristiques LBP
    lbp_features = extract_lbp_features(roi)
    
    return gabor_features, lbp_features

# Fonction pour extraire les caractéristiques de texture Gabor
def extract_gabor_features(image):
    gabor_filters = []
    for theta in np.arange(0, np.pi, np.pi / 4):
        for sigma in (1, 3):
            kernel = cv2.getGaborKernel((21, 21), sigma, theta, 10, 0.5, 0, ktype=cv2.CV_32F)
            gabor_filters.append(kernel)
    features = []
    for kernel in gabor_filters:
        filtered = cv2.filter2D(image, cv2.CV_8UC3, kernel)
        features.append(filtered.mean())
        features.append(filtered.var())
    return np.array(features)

# Fonction pour extraire les caractéristiques de texture LBP
def extract_lbp_features(image, P=8, R=1):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray_image, P, R, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, P + 3), range=(0, P + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    return hist

# Définition des caractéristiques de texture pour chaque type de pièce
texture_features = {
    "1 Euro": (
        [2.54802772e+02, 2.27790550e+01, 2.52533960e+02, 5.88788253e+02,
         2.54743333e+02, 2.47746173e+01, 2.53384818e+02, 3.96980232e+02,
         2.54990330e+02, 4.72084710e-01, 2.54878020e+02, 2.38904674e+01,
         2.54952640e+02, 3.62122240e+00, 2.54531419e+02, 1.00410069e+02],
        [0.09811881, 0.10306931, 0.03346535, 0.06930693, 0.11940594,
         0.10019802, 0.03980198, 0.10306931, 0.15970297, 0.17386139]
    ),
    "2 Euro": (
        [2.55000000e+02, 0.00000000e+00, 2.52860045e+02, 5.24797918e+02,
         2.55000000e+02, 0.00000000e+00, 2.53209494e+02, 4.28425923e+02,
         2.54983643e+02, 7.56769493e-01, 2.51685986e+02, 8.00307080e+02,
         2.55000000e+02, 0.00000000e+00, 2.53050945e+02, 4.73318251e+02],
        [0.0922449, 0.09052154, 0.03564626, 0.07047619, 0.089161,
         0.10031746, 0.06657596, 0.09251701, 0.17687075, 0.18566893]
    ),
    "5 Cent": (
        [2.55000000e+02, 0.00000000e+00, 2.54197203e+02, 1.73202535e+02,
         2.55000000e+02, 0.00000000e+00, 2.54048652e+02, 2.25685400e+02,
         2.54999885e+02, 3.90306883e-04, 2.52128507e+02, 6.73377227e+02,
         2.55000000e+02, 0.00000000e+00, 2.54531363e+02, 1.04192259e+02],
        [0.07797217, 0.09670754, 0.03602425, 0.07390825, 0.11881802,
         0.09987602, 0.06295633, 0.09973826, 0.14657666, 0.18742251]
    ),
    "10 Cent": (
        [2.54971225e+02, 1.19594642e+00, 2.53437174e+02, 3.61047648e+02,
         2.55000000e+02, 0.00000000e+00, 2.53713861e+02, 3.05744176e+02,
         2.54961421e+02, 2.16289791e+00, 2.53536740e+02, 3.39239771e+02,
         2.54998039e+02, 4.12634781e-02, 2.54038932e+02, 2.26580655e+02],
        [0.10668253, 0.10238615, 0.03529645, 0.06662701, 0.09551193,
         0.08592769, 0.04236896, 0.11197039, 0.15837134, 0.19485756]
    ),
    "20 Cent": (
        [254.93381485, 5.92137826, 253.10639628, 445.50679452, 254.77033127,
         32.19007006, 251.86570827, 765.49491802, 254.87813969, 15.06040992,
         252.07676459, 700.58984301, 254.97254439, 1.52851906, 253.35444591,
         395.98152832],
        [0.10736857, 0.10292381, 0.02944649, 0.05333704, 0.11160497,
         0.06833808, 0.03909994, 0.10410445, 0.1939718, 0.18980485]
    ),
    "50 Cent": (
        [254.98361582, 0.96564493, 253.92266949, 256.70138817, 254.92474105,
         6.50598863, 253.24576271, 432.94981726, 254.97311676, 1.29122644,
         253.32318738, 409.13502732, 255.0, 0.0, 254.10736817, 213.8944749],
        [0.10847458, 0.10621469, 0.02980226, 0.0575565, 0.1075565,
         0.07831921, 0.03820621, 0.10600282, 0.18396893, 0.18389831])

}
def classify_region(gabor_features, lbp_features, texture_features):
    min_distance = float('inf')
    closest_coin = None
    
    for coin, (coin_gabor, coin_lbp) in texture_features.items():
        # Calculer la distance euclidienne entre les caractéristiques de la région d'intérêt et celles de la pièce
        distance = np.sqrt(np.sum((gabor_features - coin_gabor) ** 2) + np.sum((lbp_features - coin_lbp) ** 2))
        
        # Vérifier si la distance calculée est la plus petite jusqu'à présent
        if distance < min_distance:
            min_distance = distance
            closest_coin = coin
    
    return closest_coin

# Charger l'image
image_path = "dataset/images/93.JPG"
image = cv2.imread(image_path)

# Extraire les régions d'intérêt
rois, _, _ = extract_ROI(image_path)

# Initialiser le dictionnaire pour sauvegarder les résultats de classification
classified_regions = {}

# Parcourir chaque région d'intérêt
for i, roi in enumerate(rois):
    # Extraire les caractéristiques de texture de la région d'intérêt
    gabor_features_roi, lbp_features_roi = extract_texture_features(roi)
    
    # Classer la région d'intérêt
    classified_coin = classify_region(gabor_features_roi, lbp_features_roi, texture_features)
    
    # Sauvegarder le résultat de classification
    classified_regions[f"Region {i+1}"] = classified_coin

# Afficher les résultats de classification pour chaque région d'intérêt
for region, coin in classified_regions.items():
    print(f"La région d'intérêt {region} appartient à la classe:", coin)

