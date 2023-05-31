import cv2
import os
from datetime import datetime
import time 
#Script de capture vidéo avec la camera
# Pictures will be stored in the the folder captures with the format name Year month day Hour minutes secondes 
def capture_image(number_pic ):
    # Créer un dossier pour enregistrer les images s'il n'existe pas déjà
    if not os.path.exists("captures"):
        os.makedirs("captures")

    # Ouvrir la webcam
    cap = cv2.VideoCapture(0)

    # Vérifier si la webcam est ouverte correctement
    if not cap.isOpened():
        print("Impossible d'ouvrir la webcam.")
        return

    while True:
        # Lire une image depuis la webcam
        ret, frame = cap.read()

        # Afficher l'image capturée
        #cv2.imshow("Webcam", frame
        # Vérifier si l'utilisateur a appuyé sur la touche 's' pour capturer l'image
        for i  in range(0,number_pic ):
            # Obtenir la date et l'heure actuelles
            now = datetime.now()
            time.sleep(0.5)
            timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")

            # Créer le nom de fichier en utilisant la date et l'heure
            filename = f"captures/capture_{timestamp}.jpg"

            # Enregistrer l'image capturée
            cv2.imwrite(filename, frame)

            print(f"Image capturée",i," : {filename}")
            i=i+1
		
        # Vérifier si l'utilisateur a appuyé sur la touche 'q' pour quitter
        break

    # Fermer la webcam et détruire les fenêtres OpenCV
    cap.release()
    cv2.destroyAllWindows()

# Appeler la fonction pour capturer des images
capture_image(40)
