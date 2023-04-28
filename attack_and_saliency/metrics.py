import numpy as np
import cv2
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity as ssim

# Ouvrir les cartes de saillance
def img_open(path_sal_map,path_attacked_sal_map):
    img1 = cv2.imread(path_sal_map)
    img2 = cv2.imread(path_attacked_sal_map)
    return img1,img2

# Calcul de la différence absolue
# Plus la valeur est proche de 0 plus les images sont similaires
def diff_abs(path_sal_map,path_attacked_sal_map):
    img1, img2 = img_open(path_sal_map,path_attacked_sal_map)
    return np.sum(np.abs(img1 - img2))

# Calcul de la différence quadratique (mse)
# Plus la valeur est proche de 0 plus les images sont similaires
def diff_quadratique(path_sal_map,path_attacked_sal_map):
    img1, img2 = img_open(path_sal_map,path_attacked_sal_map)
    return np.sum(np.square(img1 - img2))

# Calcul du coef de corrélation
# Plus la valeur est proche de 1 plus les images sont similaires
def coef_correlation(path_sal_map,path_attacked_sal_map):
    img1, img2 = img_open(path_sal_map,path_attacked_sal_map)
    result = pearsonr(img1.flatten(),img2.flatten())
    return result.statistic

# Calcul du ssim Structural Similarity Index
# Plus la valeur est proche de 1 plus les images sont similaires
def ssim_func(path_sal_map,path_attacked_sal_map):
    img1 = cv2.imread(path_sal_map)
    img2 = cv2.imread(path_attacked_sal_map)
    gray_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    valssim = ssim(gray_img1, gray_img2)
    return valssim
