print('Setting UP')
import os
from cv2 import log
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from sklearn.model_selection import train_test_split
from img_utils import *
import tensorflow as tf
# from tensorflow.keras.callbacks import CSVLogger
import threading
import shutil

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

# cv2.waitKey(0)
# print(steerings[0])

#### STEP 4 - SPLIT FOR TRAINING AND VALIDATION
xTrain, xVal, yTrain, yVal = train_test_split(imagesPath, steerings,
                                              test_size=0.2,random_state=10)
print('Total Training Images: ',len(xTrain))
print('Total Validation Images: ',len(xVal))

#### STEP 5 - AUGMENT DATA

#### STEP 6 - PREPROCESS

#### STEP 7 - CREATE MODEL
model = createModel()
model.summary()

#### STEP 8 - LAUNCH TENSORBOARD
logPath = os.path.join(os.getcwd(), 'tflog')
print(logPath)

# Clear any logs from previous runs
try:
    print(f"Removing Tensorboard log directory {logPath}")
    shutil.rmtree(logPath)
except OSError as e:
    print("Error: %s : %s" % (logPath, e.strerror))

def launchTensorBoard():
    import os
    os.system('tensorboard --logdir=' + logPath)
    return

t = threading.Thread(target=launchTensorBoard, args=([]))
t.start()

#Allow TensorBoard callbacks
BATCH_SIZE = 50
tensorboard_callback = tf.keras.callbacks.TensorBoard(logPath,
                                                      histogram_freq=1,
                                                      write_graph=True,
                                                      write_images=True)
#### STEP 9 - TRAINNING
history = model.fit(dataGen(xTrain, yTrain, 100, 1),
                            steps_per_epoch=100,
                            epochs=10,
                            validation_data=dataGen(xVal, yVal, BATCH_SIZE, 0),
                            validation_steps=50,
                            callbacks=[tensorboard_callback])

#### STEP 10 - SAVE THE KERAS H5 MODEL
model.save('model.h5')
print('Model Saved')

#### STEP 11 - PLOT THE RESULTS
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training', 'Validation'])
plt.title('Loss')
plt.xlabel('Epoch')
plt.show()