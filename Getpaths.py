# Script Python
# clement perchais 31/05/2023

import os

def get_image_paths(folder_path):
    image_paths = []
    valid_extensions = ['.jpg', '.jpeg', '.png']  # Extensions d'images valides

    # Parcours de tous les fichiers du dossier
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)

        # Vérification si le fichier a une extension d'image valide
        if os.path.isfile(file_path) and any(file_name.lower().endswith(ext) for ext in valid_extensions):
            image_paths.append(file_path)

    return image_paths

# Exemple d'utilisation
#folder_path ='/home/lisbonne/Test Python/captures'
#image_paths = get_image_paths(folder_path)

# Affichage des chemins des images
#for image_path in image_paths:
#    print(image_path)
