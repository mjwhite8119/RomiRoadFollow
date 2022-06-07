# Collect Data On the Romi

## Step 1 - Deploy the Data Collection Scripts
To deploy the data collection scripts on the Romi:

- Start up the Romi, go to the Romi Web UI
- Make the file system *Writable*.
- Go to *Application Section -> File Upload* and upload the `img_helpers.py` file.
- Go to *Application Section -> Application* and select *Uploaded Python file*.
- Upload the `uploaded.py` file.
- Go to the *Vision Status* and **Enable** the *Console Output*.
- Click the *Up* button to start the data collection application.

## Step 2 - Gather the Data

Start the MLBasic WPI Java program.

Drive the Romi around the roadway.

## Step 3 - Get data from the Raspberry Pi

To get the data from the Raspberry Pi and place it into the project *Training* directory.  On the local PC:

    cd ~/Documents/RomiRoadFollow/Training
    scp -r pi@10.0.0.2:~/DataCollected .