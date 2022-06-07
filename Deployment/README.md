# Deploy the Model on the Romi
You're now ready to deploy this to the Romi. Start up the Romi, go to the Romi Web UI

- Make the file system *Writable*.
- Go to *Application Section -> File Upload* and upload the `simple_frozen_graph.blob` file and the `wpi_helpers.py` file.
- Go to *Application Section -> Application* and select *Uploaded Python file*.
- Upload the `uploaded.py` file.
- Go to the *Vision Status* and **Enable** the *Console Output*.
- Click the *Up* button to start the road following application.

On your PC:

- Start the *BasicML* WPI progam and select the Road Following command in the AutoChooser.
- Put the robot into Automomous mode.