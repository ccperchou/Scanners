import RPi.GPIO as GPIO
import time
import cv2
from datetime import datetime

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
coil_A_1_pin = 18 # pink
coil_A_2_pin = 23 # orange
coil_B_1_pin = 24 # blue
coil_B_2_pin = 25 # yellow

# adjust if different
StepCount=8
Seq = [
    [0,1,0,0],
    [0,1,0,1],
    [0,0,0,1],
    [1,0,0,1],
    [1,0,0,0],
    [1,0,1,0],
    [0,0,1,0],
    [0,1,1,0]
    ]


GPIO.setup(coil_A_1_pin, GPIO.OUT)
GPIO.setup(coil_A_2_pin, GPIO.OUT)
GPIO.setup(coil_B_1_pin, GPIO.OUT)
GPIO.setup(coil_B_2_pin, GPIO.OUT)


def setStep(w1, w2, w3, w4):
    GPIO.output(coil_A_1_pin, w1)
    GPIO.output(coil_A_2_pin, w2)
    GPIO.output(coil_B_1_pin, w3)
    GPIO.output(coil_B_2_pin, w4)

def forward(delay, steps):
    for i in range(steps):
        for j in range(StepCount):
            setStep(Seq[j][0], Seq[j][1], Seq[j][2], Seq[j][3])
            time.sleep(delay)

def backwards(delay, steps):
    for i in range(steps):
        for j in reversed(range(StepCount)):
            setStep(Seq[j][0], Seq[j][1], Seq[j][2], Seq[j][3])
            time.sleep(delay)
def moteur_360():
        print("-----Rotation Starts-----")
        forward(int(10) / 1000.0, int(500))
        print("-----Rotation Ends------")
        print("angle : 360")
        print("------------------------------")

def rotation_180_capture():
    # Nombre total de degrés de rotation (360 pour un tour complet)
    total_degrees =180


    # Ouvrir la webcam
    camera = cv2.VideoCapture(0)

    # Vérifier si la webcam est ouverte correctement
    if not camera.isOpened():
        print("Impossible d'ouvrir la webcam.")


    # Boucle pour faire tourner le moteur et prendre une photo à chaque degré de rotation
    for degree in range(total_degrees):
        # Fait avancer le moteur d'une étape
        forward(int(5) / 1000.0, int(5))
        time.sleep(1)

        ret,frame = camera.read()
        now = datetime.now()
        time.sleep(0.5)
        timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")

        # Créer le nom de fichier en utilisant la date et l'heure
        filename = f"captures/capture_{timestamp}.jpg"

        # Enregistrer l'image capturée
        cv2.imwrite(filename, frame)

        print(f"Image capturée",degree," : {filename}")

    # Libère les ressources
    GPIO.cleanup()

rotation_180_capture()
