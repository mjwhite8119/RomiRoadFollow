# Romi Road Follow
This project implements road following on a Romi robot using a deep learning neural network.  There are three main components for this project:

- **Data Collection** - collects images and steering values on the Romi for training.
- **Training** - Uses the images and steering values to train the neural network model.
- **Deployment** - deploys the trained model on the Romi to send data back to the Java WPI program and control the robot.

The WPI Java program can be obtained from from the [RomiExamples](https://github.com/FRC-2928/RomiExamples.git) repository on Github.  The example program used is called [BasicML](https://github.com/FRC-2928/RomiExamples/tree/main/BasicML)

## Step 1 - Collect Data On the Romi

#### Step 1.3 - Get data from the Raspberry Pi

To get the data from the Raspberry Pi and place it into the project *Training* directory.  On the local PC:

    cd ~/Documents/RomiRoadFollow/Training
    scp pi@10.0.0.2:~/FRC-OAK-Deployment-Models/DataCollected .

## Step 2 - Train the Model
The step involves training the model and converting it for use on the OAK-D camera device.

#### Step 2.1 - Train the Model
 Run the following commands at the terminal to train the model.  First ensure that you are in the conda `tf-env`:

    conda activate tf-env 
    cd ~/Documents/RomiRoadFollow/Training
    python training.py

The training will bring up plots showing the distribution of the data.  Mouse over the plot and type `q` to continue.  The training will run for 10 Epocs and show another plot with the results.  Type `q` again to end the training. There will be a file called `model.h5` inside of the *Training* directory.

#### Step 2.2 - Convert the Model
The model must be converted to a format that runs on the OAK-D camera.  

    python h5_to_blob_convert.py

This will create a `simple_model` folder in the Training directory together with `simple_frozen_graph.pb` file.  It will also create the following file:

    $HOME/.cache/blobconverter/simple_frozen_graph_openvino_2021.4_4shave.blob

which is the file that will be deployed to the Romi.  you can move to the project directory as:

    cd ~/Documents/RomiRoadFollow/Deployment
    mv $HOME/.cache/blobconverter/simple_frozen_graph_openvino_2021.4_6shave.blob simple_frozen_graph.blob

## Step 3 - Deploy the Model on the Romi
You're now ready to deploy this to the Romi. Start up the Romi, go to the Romi Web UI

- Make the file system *Writable*.
- Go to *Application Section -> File Upload* and upload the `simple_frozen_graph.blob` file and the `wpi_helpers.py` file.
- Go to *Application Section -> Application* and upload the `uploaded.py` file.
- Go to the *Vision Status* and **Enable** the *Console Output*.
- Click the *Up* button to start the road following application.

On your PC:

- Start the *BasicML* WPI progam and select the Road Following command in the AutoChooser.

## Install Tensorflow on M1 Mac

    conda create --name tf-env python=3.8
    conda activate tf-env
    conda install -c apple tensorflow-deps
    pip install tensorflow-macos
    pip install tensorflow-metal
    conda install -c conda-forge -y pandas jupyter

    python3 -m pip install -r requirements.txt

## Referencies

- [Frozen graph format how-to](https://leimao.github.io/blog/Save-Load-Inference-From-TF2-Frozen-Graph/)
