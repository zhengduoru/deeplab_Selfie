import cv2
import os
import numpy as np

path1 = './data/'
path2 = './label/'
list1 = os.listdir(path1)
list2 = os.listdir(path2)
for i in list1:
    im = cv2.imread(path1 + i)
    if max(im.shape) > 512:
        im = cv2.resize(im, (int(im.shape[0] / max(im.shape) * 512), int(im.shape[1] / max(im.shape) * 512)))
    cv2.imwrite(path1 + i, im)

for i in list2:
    im = cv2.imread(path2 + i, cv2.IMREAD_GRAYSCALE)
    kernel = np.ones((3,3),dtype=np.uint8)
    im = cv2.morphologyEx(im, cv2.MORPH_OPEN, kernel)
    im = cv2.morphologyEx(im, cv2.MORPH_CLOSE, kernel)
    im = np.where(im > 125, 1, 0).astype('uint8')
    if max(im.shape) > 512:
        im = cv2.resize(im, (int(im.shape[0] / max(im.shape) * 512), int(im.shape[1] / max(im.shape) * 512)))
    cv2.imwrite(path2 + i,im)
