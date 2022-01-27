import cv2
from deepface import DeepFace
import numpy as np

imgpath = "/home/luz/Pictures/captureLuz.jpg"

analyze = DeepFace.analyze(imgpath, actions=['emotion','age', 'gender', 'race'], models={}, enforce_detection=False)
print(analyze)
