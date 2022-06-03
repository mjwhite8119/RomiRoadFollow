print('Setting UP')
import os
from tabnanny import verbose
from cv2 import log
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from sklearn.model_selection import train_test_split
from img_utils import *
from tensorboard_utils import *
import tensorflow as tf

#### STEP 1 - INITIALIZE DATA
path = 'DataCollected'
data = importDataInfo(path)
print(data.head())

#### STEP 2 - VISUALIZE AND BALANCE DATA
data = balanceData(data,display=False)

#### STEP 3 - PREPARE FOR PROCESSING
imagesPath, steerings = loadData(path,data)
print('Number of Path Created for Images ',len(imagesPath),len(steerings))

#### STEP 4 - SPLIT FOR TRAINING AND VALIDATION
xTrain, xVal, yTrain, yVal = train_test_split(imagesPath, steerings,
                                              test_size=0.2,random_state=10)
print('Total Training Images: ',len(xTrain))
print('Total Validation Images: ',len(xVal))

#### STEP 5 - LAUNCH TENSORBOARD
batch_size = 100
logPath = os.path.join(os.getcwd(), 'tflog')
tensorboard_callback = startTensorBoard(logPath)

writer = tf.summary.create_file_writer("tflog/train/")     

#### STEP 6 - DISPLAY IN TENSORBOARD
step = 0
num_epochs = 1
class_names = ["steering"]

for epoch in range(num_epochs):
    train_batch = dataGen(xTrain, yTrain, batch_size, 1)
    for batch_idx, (x, y) in enumerate(train_batch):
        figure = my_image_grid(x, y)

        with writer.as_default():
            tf.summary.image(
                "Visualize Images", plot_to_image(figure), step=step,
            )
            step += 1
