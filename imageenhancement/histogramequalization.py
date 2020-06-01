import os
import cv2

#Histogram equalizer. Using inbuilt opencv function
img_path = os.path.join(os.getcwd(), 'images/blur_colour.jpg')
img = cv2.imread(img_path)
img_binary = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_hist_enhanced = cv2.equalizeHist(img_binary)

cv2.imshow('blurred colour image', img)
cv2.imshow('blurred binary image', img_binary)
cv2.imshow('enhanced binary image', img_hist_enhanced)
cv2.waitKey(0)

