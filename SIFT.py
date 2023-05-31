import cv2
import os
from datetime import datetime
import time
import numpy as np 
# Methode Script 
# Input : Pic n°1 and Pic n° 2
# Output : Commun points and marker

# function  find_matching points for two pics
def find_matching_points(image1, image2):
    # Chargement des images
    img1 = cv2.imread(image1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image2, cv2.IMREAD_GRAYSCALE)

    # Création de l'extracteur de points d'intérêt
    sift = cv2.SIFT_create()

    # Détection des points d'intérêt et de leurs descripteurs dans les deux images
    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

    # Création du matcher de descripteurs
    matcher = cv2.BFMatcher(cv2.NORM_L2)

    # Recherche des correspondances entre les descripteurs
    matches = matcher.match(descriptors1, descriptors2)

    # Trie les correspondances en fonction de leur similarité
    matches = sorted(matches, key=lambda x: x.distance)

    # Sélectionne les meilleures correspondances (20% des correspondances avec la plus petite distance)
    num_good_matches = int(len(matches) * 0.2)
    good_matches = matches[:num_good_matches]

    # Récupère les points d'intérêt correspondants dans les deux images
    points1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    points2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    return points1, points2

def find_matching_points(images):
    # Chargement des images et création de l'extracteur de points d'intérêt
    sift = cv2.SIFT_create()
    keypoints_list = []
    descriptors_list = []

    # Extraction des points d'intérêt et des descripteurs pour chaque image
    for image in images:
        img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        keypoints, descriptors = sift.detectAndCompute(img, None)
        keypoints_list.append(keypoints)
        descriptors_list.append(descriptors)

    matcher = cv2.BFMatcher(cv2.NORM_L2)

    common_points = []
    for i in range(len(images)-1):
        # Recherche des correspondances entre les descripteurs de l'image i et l'image suivante (i+1)
        matches = matcher.match(descriptors_list[i], descriptors_list[i+1])
        matches = sorted(matches, key=lambda x: x.distance)

        # Sélection des meilleures correspondances
        num_good_matches = int(len(matches) * 0.2)
        good_matches = matches[:num_good_matches]

        # Récupération des points d'intérêt correspondants
        points1 = np.float32([keypoints_list[i][m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        points2 = np.float32([keypoints_list[i+1][m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        common_points.append((points1, points2))

    return common_points

# Exemple d'utilisation
image_paths = ['chemin/vers/image1.jpg', 'chemin/vers/image2.jpg', 'chemin/vers/image3.jpg']

common_points = find_matching_points(image_paths)

# Affichage des points d'intérêt correspondants sur les paires d'images
for i, (points1, points2) in enumerate(common_points):
    img1 = cv2.imread(image_paths[i])
    img2 = cv2.imread(image_paths[i+1])
    matched_img = cv2.drawMatches(img1, keypoints_list[i], img2, keypoints_list[i+1], good_matches, None)

    cv2.imshow('Matches', matched_img)
    cv2.waitKey(0)

cv2.destroyAllWindows()
