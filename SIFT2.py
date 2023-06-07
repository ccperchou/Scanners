# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 14:19:46 2023

@author: clement perchais 

# algorithm detection between two images 
"""

import cv2
from matplotlib import pyplot as plt
import numpy as np

 

def reconstruct_3d_structure(image1_path, image2_path):
    # Charger les images
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)

    # Convertir les images en niveaux de gris
    gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Créer l'objet détecteur SIFT
    sift = cv2.xfeatures2d.SIFT_create()

    # Détecter les points d'intérêt et extraire les descripteurs pour les deux images
    keypoints1, descriptors1 = sift.detectAndCompute(gray_image1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray_image2, None)

    # Créer l'objet FLANN Matcher (approximation rapide du plus proche voisin)
    flann = cv2.FlannBasedMatcher()

    # Trouver les correspondances entre les descripteurs des deux images
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    # Filtrer les correspondances en utilisant le ratio test de Lowe
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # Extraire les points correspondants des deux images
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Estimer la matrice fondamentale à l'aide de l'algorithme RANSAC
    _, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Sélectionner uniquement les correspondances valides
    src_pts = src_pts[mask.ravel() == 1]
    dst_pts = dst_pts[mask.ravel() == 1]

    # Estimer la matrice essentielle à partir de la matrice fondamentale
    essential_matrix, _ = cv2.findEssentialMat(src_pts, dst_pts)

    # Récupérer la pose de la caméra à partir de la matrice essentielle
    _, rotation, translation, _ = cv2.recoverPose(essential_matrix, src_pts, dst_pts)

    # Triangulation des points correspondants pour la reconstruction 3D
    projection_matrix1 = np.hstack((np.eye(3), np.zeros((3, 1))))
    projection_matrix2 = np.hstack((rotation, translation))
    points_3d = cv2.triangulatePoints(projection_matrix1, projection_matrix2, src_pts.reshape(-1, 2).T, dst_pts.reshape(-1, 2).T)

    # Convertir les coordonnées homogènes en coordonnées 3D
    points_3d /= points_3d[3]

    # Afficher les résultats
    print("Reconstructed 3D Points:")
    print(points_3d)


def estimate_camera_pose(image1_path, image2_path):
    # Charger les images
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)

    # Convertir les images en niveaux de gris
    gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Créer l'objet détecteur SIFT
    sift = cv2.xfeatures2d.SIFT_create()

    # Détecter les points d'intérêt et extraire les descripteurs pour les deux images
    keypoints1, descriptors1 = sift.detectAndCompute(gray_image1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray_image2, None)

    # Créer l'objet FLANN Matcher (approximation rapide du plus proche voisin)
    flann = cv2.FlannBasedMatcher()

    # Trouver les correspondances entre les descripteurs des deux images
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    # Filtrer les correspondances en utilisant le ratio test de Lowe
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # Extraire les points correspondants des deux images
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Estimer la matrice fondamentale à l'aide de l'algorithme RANSAC
    _, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Sélectionner uniquement les correspondances valides
    src_pts = src_pts[mask.ravel() == 1]
    dst_pts = dst_pts[mask.ravel() == 1]

    # Estimer la matrice essentielle à partir de la matrice fondamentale
    essential_matrix, _ = cv2.findEssentialMat(src_pts, dst_pts)

    # Récupérer la pose de la caméra à partir de la matrice essentielle
    _, rotation, translation, _ = cv2.recoverPose(essential_matrix, src_pts, dst_pts)

    # Afficher les résultats
    print("Rotation:")
    print(rotation)
    print("Translation:")
    print(translation)

 






def detect_common_points(image1_path, image2_path):
    # Charger les images
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)

    # Convertir les images en niveaux de gris
    gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Créer l'objet détecteur SIFT
    sift = cv2.xfeatures2d.SIFT_create()

    # Détecter les points d'intérêt et les descripteurs pour les deux images
    keypoints1, descriptors1 = sift.detectAndCompute(gray_image1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray_image2, None)

    # Créer un objet BFMatcher (Brute-Force Matcher)
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    # Faire correspondre les descripteurs des deux images
    matches = bf.match(descriptors1, descriptors2)

    # Trier les correspondances en fonction de leur distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Afficher les correspondances
    result_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches[:1000], None)

    # Afficher l'image avec les correspondances
    cv2.imshow('Common Points', result_image)
    #print(result_image)
    #cv2.imshow('image1', gray_image1)
    #cv2.imshow('image2', gray_image2)
    # print les points correpondant
    plt.imshow(result_image)
    
    cv2.destroyAllWindows()

# Chemin des deux images à comparer
image1_path = '/home/lisbonne/Test Python/captures/capture_2023-06-06_15-29-08.jpg'
image2_path = '/home/lisbonne/Test Python/captures/capture_2023-06-06_15-29-26.jpg'

# Appeler la fonction pour détecter les points communs
detect_common_points(image1_path, image2_path)
# Appeler la fonction pour estimer la pose de la caméra
estimate_camera_pose(image1_path, image2_path)
# Fonction finale 
reconstruct_3d_structure(image1_path, image2_path)
