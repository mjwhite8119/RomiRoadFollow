import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import io
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import tensorflow_datasets as tfds
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from tensorboard_utils import *
from tensorflow.keras.models import load_model
from img_utils import *

#### STEP 1 - LOAD AN IMAGE FOR MODEL INPUT
path = 'DataCollected'
data = importDataInfo(path)
print(data.head())
print(" ")

indexed_data = data.iloc[0]
print(f"Indexed data first line {indexed_data}")
print(" ")
print(f"Indexed data first column {indexed_data[0]}")

# Read the actual image
imagePath = os.path.join(path,indexed_data[0])
img =  mpimg.imread(imagePath)
img = preProcess(img)

input = np.array([img])

#### LAUNCH TENSORBOARD
logPath = os.path.join(os.getcwd(), 'tflog')
tensorboard_callback = startTensorBoard(logPath)

writer = tf.summary.create_file_writer(f"{logPath}/graph_vis")

#### DISPLAY THE MODEL 
model = createModelFunctional(img)
model.summary()

@tf.function
def my_model(x):
    return model(x)

@tf.function
def my_func(x, y):
    return tf.nn.relu(tf.matmul(x, y))

x = tf.ones((1, 200, 200, 3))
# x = tf.random.uniform((3, 3))
# y = tf.random.uniform((3, 3))

tf.summary.trace_on(graph=True, profiler=True)
# out = my_func(x, y)
out = my_model(x)

with writer.as_default():
    tf.summary.trace_export(
        name="function_trace", step=0, profiler_outdir=f"{logPath}\\graph_vis\\"
    )