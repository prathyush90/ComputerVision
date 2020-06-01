import numpy as np
import cv2
import os

img_path = os.path.join(os.getcwd(), 'images/noisy.jpeg')
img = cv2.imread(img_path)

img_enhanced = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)

cv2.imshow('blurred  image', img)
cv2.imshow('enhanced  image', img_enhanced)
cv2.waitKey(0)