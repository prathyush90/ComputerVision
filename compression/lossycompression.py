import os
import cv2
import math
import numpy as np
N = 8
from scipy.fftpack import dct, idct

#Huffman coding is not implemented. Huffman coding needs to be done after Quantization to observe size reduction
# implement 2D DCT
def dct2(a):
    return dct(dct(a.T, norm='ortho').T, norm='ortho')

# implement 2D IDCT
def idct2(a):
    return idct(idct(a.T, norm='ortho').T, norm='ortho')
def img_inverse_transform(img):

    # DCT will split image into 8*8 blocks which is followed by jpeg. Will use default

    rows, cols = img.shape
    # ceiling will remove some data better is floow
    NR = math.floor(rows / N)
    NC = math.floor(cols / N)

    image = np.zeros(img.shape)
    for i in range(1, NR):
        for j in range(1, NC):
            subm = img[N * (i - 1): (N * i), N * (j - 1): (N * j)]
            out = idct2(subm)
            image[N * (i - 1): (N * i), (N * (j - 1)): (N * j)] = out
    print("inverse")
    print(image)
    image = np.uint8(np.round(image))
    return image

def img_transform(img):

    print("before transform")
    print(img)

    #DCT will split image into 8*8 blocks which is followed by jpeg. Will use default

    rows,cols = img.shape
    #ceiling will remove some data better is floow
    NR = math.floor(rows/N)
    NC = math.floor(cols/N)

    T = np.zeros(img.shape)
    #Apply default DCT opencv
    for i in range(1, NR):
        for j in range(1, NC):
            r = (N*(i-1),(N*i))
            c = (N*(j-1), (N*j))
            subm = img[N*(i-1) : (N*i), N*(j-1) : (N*j)]
            out  = dct2(subm)
            T[N * (i - 1): (N * i), (N * (j - 1)): (N * j)] = out
    print("transform")
    print(T)
    return T

def img_quantize(img):

    rows, cols = img.shape
    NR = math.floor(rows / N)
    NC = math.floor(cols / N)
    Q = np.zeros(img.shape)

    Qm = [
            [16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77],
            [24, 35, 55, 64, 81, 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 99]
          ]
    for r in range(1, NR):
        for c in range(1, NC):
            subm = np.float64(img[N * (r - 1): (N * r),  N * (c - 1): (N * c)])
            Q[N * (r - 1): (N * r),  N * (c - 1): (N * c)]  = np.multiply(np.round(subm/ Qm), Qm)

    return Q


def image_colorCompress(img):
    #imag should be bgr format
    Y =  cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    C = Y * 0
    C[:,:, 0] = img_inverse_transform(img_quantize(img_transform(Y[:,:, 0])))
    C[:, :, 1] = img_inverse_transform(img_quantize(img_transform(Y[:, :, 1])))
    C[:, :, 2] = img_inverse_transform(img_quantize(img_transform(Y[:, :, 2])))
    #C = cv2.cvtColor(C, cv2.COLOR_YCR_CB2BGR)
    return cv2.cvtColor(C, cv2.COLOR_YCR_CB2BGR)

img_path = os.path.join(os.getcwd(), 'data/testimag.jpeg')
img = cv2.imread(img_path)
#get the image and show it in opencv window
cv2.imshow('uncompressedimage', img)
img_compressed = image_colorCompress(img)
cv2.imshow('compressedimage', img_compressed)

cv2.waitKey(0)
