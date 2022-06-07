#  Train the Model
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

which is the file that will be deployed to the Romi.  The conversion script will move this file into the *Deployment* directory.