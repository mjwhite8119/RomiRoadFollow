import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import cv2
import glob
import PIL.Image

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D,Flatten,Dense
from tensorflow.keras.optimizers import Adam, SGD

import matplotlib.image as mpimg
from imgaug import augmenters as iaa

import random


#### STEP 1 - INITIALIZE DATA
def getName(filePath):
    myImagePathL = filePath.split('/')[-2:]
    myImagePath = os.path.join(myImagePathL[0],myImagePathL[1])
    return myImagePath

def importDataInfo(path):
    columns = ['Center','xaxisSpeed', 'zaxisRotate']
    noOfFolders = len(os.listdir(path))//2
    data = pd.DataFrame()
    for x in range(0,1):
        print(os.path.join(path, f'log_{x}.csv'))
        dataNew = pd.read_csv(os.path.join(path, f'log_{x}.csv'), names = columns)
        print(f'{x}:{dataNew.shape[0]} ',end='')
        #### REMOVE FILE PATH AND GET ONLY FILE NAME
        #print(getName(data['center'][0]))
        dataNew['Center']=dataNew['Center'].apply(getName)
        data =data.append(dataNew,True )
    print(' ')
    print('Total Images Imported',data.shape[0])
    return data

#### STEP 2 - VISUALIZE AND BALANCE DATA
def balanceData(data,display=True):
    nBin = 31
    samplesPerBin =  50
    hist, bins = np.histogram(data['zaxisRotate'], nBin)
    if display:
        center = (bins[:-1] + bins[1:]) * 0.5
        plt.bar(center, hist, width=0.03)
        plt.plot((np.min(data['zaxisRotate']), np.max(data['zaxisRotate'])), (samplesPerBin, samplesPerBin))
        plt.title('Data Visualisation')
        plt.xlabel('zaxisRotate Angle')
        plt.ylabel('No of Samples')
        plt.show()
    removeindexList = []

    for j in range(nBin):
        binDataList = []
        for i in range(len(data['zaxisRotate'])):
            zaxis = data['zaxisRotate'][i]
            xaxis = data['xaxisSpeed'][i]
            # Balance data and only use images where robot is moving
            if zaxis >= bins[j] and zaxis <= bins[j + 1] or xaxis < 0.4:
                binDataList.append(i)
        binDataList = shuffle(binDataList)
        binDataList = binDataList[samplesPerBin:]
        removeindexList.extend(binDataList)

    print('Removed Images:', len(removeindexList))
    data.drop(data.index[removeindexList], inplace=True)
    print('Remaining Images:', len(data))
    if display:
        hist, _ = np.histogram(data['zaxisRotate'], (nBin))
        plt.bar(center, hist, width=0.03)
        plt.plot((np.min(data['zaxisRotate']), np.max(data['zaxisRotate'])), (samplesPerBin, samplesPerBin))
        plt.title('Balanced Data')
        plt.xlabel('zaxisRotate Angle')
        plt.ylabel('No of Samples')
        plt.show()
    return data

def on_press(event):
    print('press', event.key)
    sys.stdout.flush()
    if event.key == 'x':
        print("Image deleted")

def draw_image_with_label(path, data):
    for i in range(0, len(data), 1):
        indexed_data = data.iloc[i]
        imagePath = os.path.join(path,indexed_data[0])
        label = np.array([indexed_data[1], indexed_data[2]])
        color = (255, 255, 255)
        print('Actual Steering Angle = {0}'.format(label))
        print(imagePath)
        # img = mpimg.imread(imagePath)
        img, steering = augmentImage(imagePath, label)
        img = preProcess(img)
        cv2.putText(img, f"{label[1]}", 
                    (2, img.shape[0] - 4), 
                    cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
        # fig.canvas.mpl_connect('key_press_event', on_press)            
        plt.imshow(img)
        plt.show()

def saveBalancedData(path, data):
    imgList = []
    speedList = []
    rotateList = []
    for i in range(0, len(data), 1):
        indexed_data = data.iloc[i]
        imagePath = os.path.join(path,indexed_data[0])
        speed = np.array([indexed_data[1], indexed_data[1]])
        rotate = np.array([indexed_data[1], indexed_data[2]])
        imgList.append(imagePath)
        speedList.append(speed)
        rotateList.append(rotate)
        
    rawData = {'Image': imgList,
                'speed': speedList,
                'rotate': rotateList}
    df = pd.DataFrame(rawData)
    df.to_csv(os.path.join("BalancedData",f'log_{str(0)}.csv'), index=False, header=False)
    print('Balanced Data Saved')
    print('Total Images: ',len(imgList))        

#### STEP 3 - PREPARE FOR PROCESSING
def loadData(path, data):
  imagesPath = []
  steerings = []
  for i in range(len(data)):
    indexed_data = data.iloc[i]
    imagesPath.append( os.path.join(path,indexed_data[0]))
    steerings.append(np.array([indexed_data[1], indexed_data[2]]))
  imagesPath = np.asarray(imagesPath)
  steerings = np.asarray(steerings)
  return imagesPath, steerings


#### STEP 5 - AUGMENT DATA
def augmentImage(imgPath,steering):
    img =  mpimg.imread(imgPath)
    if np.random.rand() < 0.5:
        pan = iaa.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)})
        img = pan.augment_image(img)
    if np.random.rand() < 0.5:
        zoom = iaa.Affine(scale=(1, 1.2))
        img = zoom.augment_image(img)
    if np.random.rand() < 0.5:
        brightness = iaa.Multiply((0.5, 1.2))
        img = brightness.augment_image(img)
    # if np.random.rand() < 0.5:
    #     img = cv2.flip(img, 1)
    #     steering = steering[0][1] * -1
    return img, steering

# imgRe,st = augmentImage('DataCollected/IMG18/Image_1601839810289305.jpg',0)
# #mpimg.imsave('Result.jpg',imgRe)
# plt.imshow(imgRe)
# plt.show()

#### STEP 6 - PREPROCESS
def preProcess(img):
    # img = img[54:120,:,:]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img,  (3, 3), 0)
    img = cv2.resize(img, (200, 200))
    img = img/255
    return img

# imgRe = preProcess(mpimg.imread('DataCollected/IMG18/Image_1601839810289305.jpg'))
# # mpimg.imsave('Result.jpg',imgRe)
# plt.imshow(imgRe)
# plt.show()

#### STEP 7 - CREATE MODEL
def createModel():
    model = Sequential()

    model.add(Convolution2D(24, (5, 5), (2, 2), input_shape=(200, 200, 3), activation='elu'))
    model.add(Convolution2D(36, (5, 5), (2, 2), activation='elu'))
    model.add(Convolution2D(48, (5, 5), (2, 2), activation='elu'))
    model.add(Convolution2D(64, (3, 3), activation='elu'))
    model.add(Convolution2D(64, (3, 3), activation='elu'))

    model.add(Flatten())
    model.add(Dense(100, activation = 'elu'))
    model.add(Dense(50, activation = 'elu'))
    model.add(Dense(10, activation = 'elu'))
    model.add(Dense(1))

    # Compile model with optimizer
    model.compile(Adam(lr=0.0001),loss='mse')
    return model

#### STEP 8 - TRAINNING
def dataGen(imagesPath, steeringList, batchSize, trainFlag):
    while True:
        imgBatch = []
        steeringBatch = []

        for i in range(batchSize):
            index = random.randint(0, len(imagesPath) - 1)
            if trainFlag:
                img, steering = augmentImage(imagesPath[index], steeringList[index])
            else:
                img = mpimg.imread(imagesPath[index])
                steering = steeringList[index][1]
            img = preProcess(img)
            imgBatch.append(img)
            steeringBatch.append(steering)
        yield (np.asarray(imgBatch),np.asarray(steeringBatch))
