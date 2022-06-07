print('Setting UP')
import os
from tabnanny import verbose
from cv2 import log
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from sklearn.model_selection import train_test_split
from img_utils import *
import tensorflow as tf
from tensorboard_utils import *

AUTOTUNE = tf.data.experimental.AUTOTUNE

#### STEP 1 - INITIALIZE DATA
path = 'DataCollected'
data = importDataInfo(path)
print(data.head())


#### STEP 2 - VISUALIZE AND BALANCE DATA
data = balanceData(data,display=True)

# View each image
# draw_image_with_label(path, data)

#### STEP 3 - PREPARE FOR PROCESSING
imagesPath, steerings = loadData(path,data)
print('Number of Path Created for Images ',len(imagesPath),len(steerings))

#### STEP 4 - SPLIT FOR TRAINING AND VALIDATION
xTrain, xVal, yTrain, yVal = train_test_split(imagesPath, steerings,
                                              test_size=0.2,random_state=10)
print('Total Training Images: ',len(xTrain))
print('Total Validation Images: ',len(xVal))

#### STEP 5 - AUGMENT DATA

#### STEP 6 - PREPROCESS

#### STEP 7 - CREATE MODEL

# Read the actual image
indexed_data = data.iloc[0]
imagePath = os.path.join(path,indexed_data[0])
img =  mpimg.imread(imagePath)
img = preProcess(img)

# Use it for the model input shape
model = createModelFunctional(img)
# model = createRFModel(xTrain)
model.summary()

#### STEP 8 - LAUNCH TENSORBOARD
batch_size = 100
logPath = os.path.join(os.getcwd(), 'tflog')
tensorboard_callback = startTensorBoard(logPath)   

#### STEP 9 - TRAINNING
training_batch_size = 100
validation_batch_size = 50
trainSteps = int(len(xTrain) / training_batch_size) + 1
valSteps = int(len(xVal) / validation_batch_size) + 1
history = model.fit(dataGen(xTrain, yTrain, training_batch_size, 1),
                            steps_per_epoch=trainSteps,
                            epochs=10,
                            validation_data=dataGen(xVal, yVal, validation_batch_size, 0),
                            validation_steps=valSteps,
                            callbacks=[tensorboard_callback],
                            verbose=2)

#### STEP 10 - SAVE THE KERAS H5 MODEL
model.save('model.h5')
print('Model Saved')

#### STEP 11 - PLOT THE RESULTS
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.legend(['Training', 'Validation'])
# plt.title('Loss')
# plt.xlabel('Epoch')
# plt.show()