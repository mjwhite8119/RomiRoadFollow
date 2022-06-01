print('Setting UP')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from sklearn.model_selection import train_test_split
from img_utils import *

#### STEP 1 - INITIALIZE DATA
path = 'DataCollected'
data = importDataInfo(path)
print(data.head())


#### STEP 2 - VISUALIZE AND BALANCE DATA
data = balanceData(data,display=True)

# View each image
# for i in range(0, len(data), 1):
#     indexed_data = data.iloc[i]
#     imagePath = os.path.join(path,indexed_data[0])
#     label = np.array([indexed_data[1], indexed_data[2]])
draw_image_with_label(path, data)

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

#### STEP 8 - TRAINNING
history = model.fit(dataGen(xTrain, yTrain, 100, 1),
                            steps_per_epoch=100,
                            epochs=10,
                            validation_data=dataGen(xVal, yVal, 50, 0),
                            validation_steps=50)

#### STEP 9 - SAVE THE KERAS H5 MODEL
model.save('model.h5')
print('Model Saved')

#### STEP 11 - PLOT THE RESULTS
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training', 'Validation'])
plt.title('Loss')
plt.xlabel('Epoch')
plt.show()