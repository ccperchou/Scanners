# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 14:19:46 2023

@author: clement perchais 

# algorithm detection between two images 
"""

import cv2
from matplotlib import pyplot as plt
import numpy as np
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
 
 
def plot_points_3d(points_3d):
    # Créer la figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Afficher les points 3D
    ax.plot3D(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], 'o')

    # Ajouter des axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Afficher le graphique
    #plt.show()
    
    
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
    return points_3d


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
    print("essential matrix", essential_matrix)
    # Récupérer la pose de la caméra à partir de la matrice essentielle
    _, rotation, translation, _ = cv2.recoverPose(essential_matrix, src_pts, dst_pts)

    # Afficher les résultats
    print("Rotation:")
    print(rotation)
    print("Translation:")
    print(translation)

 
def save_points_to_csv(points, file_path):
    with open(file_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['X', 'Y', 'Z'])  # Écrire l'en-tête du fichier CSV
        # CSV Format { X , Y , Z}
        for i in range(points.shape[1]):
            x = points[0, i]
            y = points[1, i]
            z = points[2, i]
            writer.writerow([x, y, z])


def visualize_points(points):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Extraire les coordonnées x, y et z des points
    x = points[0]
    y = points[1]
    z = points[2]

    # Tracer les points 3D
    ax.scatter(x, y, z, c='blue', marker='o')

    # Configurer les axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Afficher la figure
    #plt.show()

def find_common_points(images):
    # Créer un détecteur de points d'intérêt
    sift = cv2.xfeatures2d.SIFT_create()    
    #sift = cv2.SIFT_create()

    # Stocker les keypoints et les descripteurs pour chaque image
    keypoints_list = []
    descriptors_list = []

    # Détecter les keypoints et calculer les descripteurs pour chaque image
    for image in images:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        keypoints_list.append(keypoints)
        descriptors_list.append(descriptors)

    # Trouver les correspondances entre les images
    bf = cv2.BFMatcher()
    matches_list = []

    for i in range(len(descriptors_list) - 1):
        matches = bf.knnMatch(descriptors_list[i], descriptors_list[i + 1], k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
        matches_list.append(good_matches)

    # Extraire les points correspondants pour chaque paire d'images
    points_2d_list = []
    for i in range(len(matches_list)):
        points1 = np.float32([keypoints_list[i][m.queryIdx].pt for m in matches_list[i]]).reshape(-1, 1, 2)
        points2 = np.float32([keypoints_list[i + 1][m.trainIdx].pt for m in matches_list[i]]).reshape(-1, 1, 2)
        points_2d_list.append((points1, points2))

    return points_2d_list
# #---
# def detect_common_points(image1_path, image2_path):
#     # Charger les images
#     image1 = cv2.imread(image1_path)
#     image2 = cv2.imread(image2_path)

#     # Convertir les images en niveaux de gris
#     gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
#     gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

#     # Créer l'objet détecteur SIFT
#     sift = cv2.xfeatures2d.SIFT_create()

#     # Détecter les points d'intérêt et les descripteurs pour les deux images
#     keypoints1, descriptors1 = sift.detectAndCompute(gray_image1, None)
#     keypoints2, descriptors2 = sift.detectAndCompute(gray_image2, None)

#     # Créer un objet BFMatcher (Brute-Force Matcher)
#     bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

#     # Faire correspondre les descripteurs des deux images
#     matches = bf.match(descriptors1, descriptors2)

#     # Trier les correspondances en fonction de leur distance
#     matches = sorted(matches, key=lambda x: x.distance)

#     # Afficher les correspondances
#     result_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches[:1000], None)

#     # Afficher l'image avec les correspondances
#     cv2.imshow('Common Points', result_image)
#     #print(result_image)
#     #cv2.imshow('image1', gray_image1)
#     #cv2.imshow('image2', gray_image2)
#     # print les points correpondant
#     plt.imshow(result_image)
    
#     cv2.destroyAllWindows()


# Test part and function call 
# Pictures path
# image1_path = 'image1.jpg'
# image2_path = 'image2.jpg'
# image3_path = 'image3.jpg'
# image4_path = 'image4.jpg'


# image1 = cv2.imread('image1.jpg')
# image2 = cv2.imread('image2.jpg')
# image3 = cv2.imread('image3.jpg')
# image4 = cv2.imread('image4.jpg')
# image5 = cv2.imread('image5.jpg')
# image6 = cv2.imread('image6.jpg')
# image7 = cv2.imread('image7.jpg')
# image8 = cv2.imread('image8.jpg')

# images = [image1, image2, image3,image4,image5,image6,image7,image8]

# # Appeler la fonction pour détecter les points communs
# ptn2D=find_common_points(images)
# # Appeler la fonction pour estimer la pose de la caméra
# estimate_camera_pose(image1_path, image2_path)
# # Fonction finale 
# ptns3D=reconstruct_3d_structure(image1_path, image2_path)
 
# #☺print csv points
# save_points_to_csv(ptns3D, "points3D.csv")

# visualize_points(ptns3D)
#plot_points_3d(ptns3D)
