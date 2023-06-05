# Scanners

Projet de Scanner de plante 

Ce projet intitulé Scanner de plante avec une rasperry à pour objectif d’obtenir un maillage 3D  d’une plante à partir d’un scanner rasperry. Ce projet se décompose en différentes étapes que sont l’acquisition d’image, la reconstruction du modèle 3D à partir d’un modèle 2D à partir d’une solution de points d’intérêt.  L’ensemble du projet est codé en python en utilisant des bibliothèques tel que OpenCV, Panda, SIFT..


Présentation du projet (FR):

Ce projet intitulé Scanner de plante avec une rasperry à pour objectif d’obtenir un maillage 3D  d’une plante à partir d’un scanner rasperry. Ce projet se décompose en différentes étapes : 

1. **Acquisition des images** : Prenez plusieurs photos d'un objet sous différents angles en veillant à couvrir tous les côtés de l'objet. Assurez-vous d'avoir une bonne luminosité et une mise au point nette.
2. **Détection des points d'intérêt** : Utilisez des algorithmes de détection de points d'intérêt tels que SIFT, SURF ou ORB pour extraire des points clés des images. Ces points d'intérêt serviront de points de correspondance entre les images.
3. **Correspondance des points** : Utilisez des descripteurs (comme les descripteurs SIFT) pour comparer les points d'intérêt extraits des différentes images et trouver des correspondances entre eux.
4. **Estimation de la pose de la caméra** : À l'aide des correspondances de points, estimez la pose de la caméra (position et orientation) pour chaque paire d'images voisines.
5. **Reconstruction de la structure 3D** : Utilisez les poses de la caméra estimées et les correspondances de points pour reconstruire la structure 3D de l'objet. Des techniques telles que la méthode de la triangulation peuvent être utilisées pour calculer la position 3D des points.
6. **Rafinement** de la reconstruction : Effectuez des étapes supplémentaires pour améliorer la qualité de la reconstruction 3D, telles que la suppression des points aberrants, la fusion des points correspondants provenant de différentes images, et la réduction du bruit.
7. **Texturisation** : Si vous disposez également d'informations de texture provenant des images, vous pouvez projeter les images sur la surface reconstruite pour créer une représentation texturée du modèle 3D.

# Deposit Git:


Lien : https://github.com/ccperchou/Scanners

Gits Commands: 
echo "# Scanners" >> README.md
git init
git add README.md
git commit -m "first commit"
git branch -M main
git remote add origin https://github.com/ccperchou/Scanners.gitgit push -u origin main

# Documentations :

Links : 

Motor and raspberry : 

https://abra-electronics.com/electromechanical/motors/stepper-motors/mini-stepper-motors/mot-28byj48-stepper-motor-w-uln2003-driver.html?sl=fr

https://ben.akrin.com/driving-a-28byj-48-stepper-motor-uln2003-driver-with-a-raspberry-pi/

https://tutorials-raspberrypi.com/how-to-control-a-stepper-motor-with-raspberry-pi-and-l293d-uln2003a/

Github issues : 

source :  https://levelup.gitconnected.com/fix-password-authentication-github-3395e579ce74































or create a new repository on the command line

echo "# Scanners" >> README.md
git init
git add README.md
git commit -m "first commit"
git branch -M main
git remote add origin git@github.com:ccperchou/Scanners.git
git push -u origin main

…or push an existing repository from the command line

git remote add origin git@github.com:ccperchou/Scanners.git
git branch -M main
git push -u origin main

…or import code from another repository

You can initialize this repository with code from a Subversion, Mercurial, or TFS project.
