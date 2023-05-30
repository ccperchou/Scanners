
# Script Python 
# clement perchais 30/5/2023
# input : Delay and number of step 
#Test results:  time delay:10 & steps forward 500
import RPi.GPIO as GPIO
import time

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


def moteur_180():
	print(" -----Rotation Starts -----")
	forward(int(10) / 1000.0, int(250))
	print("-----Rotation Ends-------")
	print("angle :180  ")
	print("------------------------------")


if __name__ == '__main__':
    while True:
        delay = input("Time Delay (ms)?")
        steps = input("How many steps forward? ")
        forward(int(delay) / 1000.0, int(steps))
        
