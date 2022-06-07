#!/usr/bin/env python3
###########################################################
# uploaded.py
# This script collects images and steering data on the Romi
###########################################################

import cv2
import argparse
import img_helpers as img
import depthai as dai
from wpi_helpers import ConfigParser, WPINetworkTables

def parse_args():
    """Parse input arguments."""
    desc = ('Capture and display live camera video, while doing '
            'real-time object detection with OpenVINO optimized '
            'YOLO model')
    parser = argparse.ArgumentParser(description=desc)
    # parser = add_camera_args(parser)
    parser.add_argument(
        '-g', '--gui', action='store_true',
        help='use desktop gui for display [False]')
    parser.set_defaults(gui=False) 
    parser.add_argument(
        '-n', '--no_network_tables', action='store_true',
        help='use WPI Network Tables [True]')
    parser.set_defaults(no_network_tables=False)   
    parser.add_argument(
        '-p', '--mjpeg_port', type=int, default=8080,
        help='MJPEG server port [8080]')    
    args = parser.parse_args()
    return args

# -------------------------------------------------------------------------
# Main Program Start
# -------------------------------------------------------------------------
def main(args, frc_config):
    # Create pipeline
    pipeline = dai.Pipeline()

    # Define source and outputs
    camRgb = pipeline.create(dai.node.ColorCamera)
    # xoutVideo = pipeline.create(dai.node.XLinkOut)
    xoutPreview = pipeline.create(dai.node.XLinkOut)
    # xoutVideo.setStreamName("video")
    xoutPreview.setStreamName("preview")

    # Properties
    camRgb.setPreviewSize(200, 200)
    camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    camRgb.setInterleaved(False)
    camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)

    # Linking
    # camRgb.video.link(xoutVideo.input)
    camRgb.preview.link(xoutPreview.input)

    # Start the mjpeg server (default)
    try:
        import cscore as cs
        mjpeg_port = 8080
        cvSource = cs.CvSource("cvsource", cs.VideoMode.PixelFormat.kMJPEG, 320, 240, 30)
        mjpeg_server = cs.MjpegServer("httpserver", mjpeg_port)
        mjpeg_server.setSource(cvSource)
        print('MJPEG server started on port', mjpeg_port)
    except Exception as e:
        cvSource = False

    print("Using Network Tables")
    networkTables = WPINetworkTables(frc_config.team)    
        
    # Connect to device and start pipeline
    with dai.Device(pipeline) as device:

        # video = device.getOutputQueue('video')
        preview = device.getOutputQueue('preview')
        waitOnce = True
        haveData = False
        waitMessage = "Waiting for Input..."

        try:
            while True:
                # videoFrame = video.get()
                previewFrame = preview.get()
                speed, rotate = networkTables.get_drive_data()
                if speed > 0.3:
                    # Only save data if the robot is moving
                    img.saveData(previewFrame.getFrame(), speed, rotate)
                    haveData = True
                    
                elif speed < 0.1 and haveData:
                    # Robot stopped moving so 
                    print("Saving log file")  
                    img.saveLog()  
                    haveData = False 
                    waitOnce = True

                elif waitOnce:
                    print(waitMessage)  
                    waitOnce = False  

                frame = previewFrame.getFrame()
                if cvSource is False:
                    # Display stream to desktop window
                    cv2.imshow("preview", frame)
                else:               
                    # Display stream to browser
                    cvSource.putFrame(frame)   

                if cv2.waitKey(1) == ord('q'):
                    break

        except KeyboardInterrupt:
            # Keyboard interrupt (Ctrl + C) detected
            pass

        print("Saving log file")  
        img.saveLog()   

if __name__ == '__main__':
    print("Running record_images.py")
    args = parse_args()

    # Load the FRC configuration file
    frc_config = ConfigParser()

    main(args, frc_config)               