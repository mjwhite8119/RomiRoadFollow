# Romi Road Follow
This project implements road following on a Romi robot using a deep learning neural network.  There are three main components for this project:

- **Data Collection** - collects images and steering values on the Romi for training. See the [DataCollection](DataCollection/README.md) README.

- **Training** - Uses the images and steering values to train the neural network model. See the [Training](Training/README.md) README.

- **Deployment** - deploys the trained model on the Romi to send data back to the Java WPI program and control the robot. See the [Deployment](Deployment/README.md) README.

The WPI Java program can be obtained from from the [RomiExamples](https://github.com/FRC-2928/RomiExamples.git) repository on Github.  The example program used is called [BasicML](https://github.com/FRC-2928/RomiExamples/tree/main/BasicML)

## Install Tensorflow on M1 Mac

    conda create --name tf-env python=3.8
    conda activate tf-env
    conda install -c apple tensorflow-deps
    pip install tensorflow-macos
    pip install tensorflow-metal
    conda install -c conda-forge -y pandas jupyter

    python3 -m pip install -r requirements.txt

Tensorboard needs a lower version of markdown, therefore:

    pip uninstall markdown
    pip install markdown==3.1.1


### LabelImg Install
Using Conda. Create a virtual environment in conda and activate it:

    conda create -n venv
    conda activate venv
Install pyqt using conda:

    conda install pyqt
Install lxml using pip:

    pip install lxml

Clone labelImg:

    git clone https://github.com/tzutalin/labelImg.git   
    cd labelImg
    make qt5py3

Run LabelImg:

    python labelImg.py


## Referencies

- [Tensorflow Basics](https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/TensorFlow/Basics)

- [Frozen graph format how-to](https://leimao.github.io/blog/Save-Load-Inference-From-TF2-Frozen-Graph/)

- [Depthai OpenCV AI Kit](https://learnopencv.com/depthai-pipeline-overview-creating-a-complex-pipeline/)
