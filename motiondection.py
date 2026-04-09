import RPi.GPIO as GPIO
import time

class detect:
  def __init__():
     pass
  def Threadmotiondetect():
    PIN = 26
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(PIN, GPIO.IN)

    try:
        while True:
            if GPIO.input(PIN):
                print("Motion Detected!")
            else:
                print("No Motion")

            time.sleep(0.5)

    except KeyboardInterrupt:
        GPIO.cleanup()