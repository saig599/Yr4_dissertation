import glob
import cv2
import os
import numpy as np

alpha = 0.5
beta = 20
# pre processing the image patches
for filename in glob.glob("./train/*.jpg"):
    print(filename)
    img = cv2.imread(filename)

    rl = cv2.resize(img,(200,64), interpolation=cv2.INTER_CUBIC) #, interpolation=cv2.INTER_CUBIC
    gray_image = cv2.cvtColor(rl, cv2.COLOR_BGR2GRAY)

    ker = np.ones((2,2), np.uint8)/3

    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    kernel = kernel /3
    cv2.imwrite(f'./coco-text_dataset/train_processed/{os.path.basename(filename)}', gray_image)


