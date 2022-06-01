from PIL import Image, ImageDraw
import keras.backend as K
from keras.preprocessing import image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import os
import numpy as np
from img_utils import importDataInfo, balanceData

def draw_image_with_label(imagePath, label, prediction=None):
    color = (255, 255, 255)
    print('Actual Steering Angle = {0}'.format(label))
    print(imagePath)
    img = mpimg.imread(imagePath)
    cv2.putText(img, f"{label[1]}", 
                (2, img.shape[0] - 4), 
                cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
    plt.imshow(img)
    plt.show()

path = 'DataCollected'
data = importDataInfo(path)

data = balanceData(data,display=False)
print(data.head())
# View each image
for i in range(0, len(data), 1):
    indexed_data = data.iloc[i]
    imagePath = os.path.join(path,indexed_data[0])
    label = np.array([indexed_data[1], indexed_data[2]])
    draw_image_with_label(imagePath, label)