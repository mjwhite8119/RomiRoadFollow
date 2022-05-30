# Romi Road Follow
This project implements road following on a Romi robot using a deep learning neural network.  There are three main components for this project:

- Data Collection - collects images and steering values on the Romi for training.
- Training - Uses the images and steering values to train the neural network model.
- Deployment - deploys the trained model on the Romi to send data back to the Java WPI program and control the robot.

The WPI Java program can be obtained from from the [RomiExamples](https://github.com/FRC-2928/RomiExamples.git) repository on Github.  The example program used is called [BasicML](https://github.com/FRC-2928/RomiExamples/tree/main/BasicML)

## Install Tensorflow on M1 Mac

    conda create --name tf-env python=3.8
    conda activate tf-env
    conda install -c apple tensorflow-deps
    pip install tensorflow-macos
    pip install tensorflow-metal
    conda install -c conda-forge -y pandas jupyter

    python3 -m pip install -r requirements.txt