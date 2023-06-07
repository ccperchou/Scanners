# Script Python
# clement perchais 30/5/

import Moteur
import Getpaths
import SIFT
import Rotation_Capture

if __name__ == '__main__':
	# Test init rotation 180 and 360
	#Moteur.moteur_180()
	#Moteur.moteur_360()
	# pictures
	#Rotation_Capture.rotation_180_capture()
	# get pictures paths
	listpath= Getpaths.get_image_paths('/home/lisbonne/Test Python/captures')
	print(listpath)

	# SIFT commun points calcul  
	commun_points=SIFT.find_matching_points(listpath)
	print(commun_points)
	

