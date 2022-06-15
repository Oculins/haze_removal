import cv2
import math
import numpy as np

def DarkChannelGetter(img, patch_size):
    [height, width] = img.shape[:2]

    # min of the 3 channels
    b, g, r = cv2.split(img)
    h = cv2.min(cv2.min(r, g), b)

    d = np.ones((height, width))
    # min of the local patch
    for x in range(height):
        for y in range(width):
            for i in range(max(0, x - patch_size), min(height, x + patch_size)):
                for j in range(max(0, y - patch_size), min(width, y + patch_size)):
                    if h[i, j] < d[i, j]:
                        d[i, j] = h[i, j]

    return d

def AtmLight(img, dark):
    [h, w] = img.shape[:2]
    size = h * w

    # the 0.1% brightest pixels
    num_brightest = int(max(math.floor(size/1000), 1))
    dark_channel = dark.reshape(size)
    image = img.reshape(size, 3)
    indices = dark_channel.argsort()
    indices = indices[size - num_brightest::]

    # A is the average of the 0.1% pixels
    atm = np.zeros([1, 3])
    for ind in range(1, num_brightest):
       atm = atm + image[indices[ind]]
    A = atm / num_brightest

    return A

def Transmission(img, A, size):
    w = 0.95
    image = np.zeros(img.shape, img.dtype)

    for i in range(0, 3):
        image[:, :, i] = img[:, :, i]/A[0, i]

    t = 1 - w * DarkChannelGetter(image, size)

    return t

def Recover(img, t, A, typical):
    recover = np.zeros(img.shape, img.dtype)
    t = cv2.max(t, typical)

    for i in range(3):
        recover[:, :, i] = (img[:, :, i] - A[0, i])/t + A[0, i]

    return recover

# input and output
index = '60'
hazy_path = './Dense_Haze_NTIRE19/hazy/' + index + '_hazy.png'
GT_path = './Dense_Haze_NTIRE19/GT/' + index + '_GT.png'
output_path = './result/'

# parameters
width, height = 320, 240
patch_size = 5
typical_t = 0.1

# read image
input_hazy = cv2.imread(hazy_path)
input_GT = cv2.imread(GT_path)
img_hazy = cv2.resize(input_hazy, (width, height))
img_GT = cv2.resize(input_GT, (width, height))

# haze removal
I = img_hazy.astype('float64')/255
dark_channel = DarkChannelGetter(I, patch_size)
A = AtmLight(I, dark_channel)
t = Transmission(I, A, patch_size)
J = Recover(I, t, A, typical_t)

# show and write
'''
cv2.imshow("dark channel", dark_channel)
cv2.imshow("t", t)
cv2.imshow('I', img_hazy)
cv2.imshow('J', J)
'''
cv2.imwrite(output_path + index + '_hazy.png', img_hazy)
cv2.imwrite(output_path + index + '_GT.png', img_GT)
cv2.imwrite(output_path + index + '_dark channel0.png', dark_channel * 255)
cv2.imwrite(output_path + index + '_result0.png', J * 255)
cv2.waitKey()
    
