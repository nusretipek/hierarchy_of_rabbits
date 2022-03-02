# Import statements

import heat_map_module
import cv2
import numpy as np

import os
from os.path import exists
import glob
import json
import time
import multiprocessing as mp

# Dense optical flow generation function


def calculate_action_dense_optical_flow(vid, crop_parameters, vid_name, location):
    start_time = time.time()

    # Global parameters and video capture
    frame_skip_constant = 25
    counter = 1
    temp_total_mag = 0
    total_distance_list = []
    cap = cv2.VideoCapture(vid)
    cap.set(1, 0)

    # Take first frame and find corners in it
    ret, old_frame = cap.read()
    old_frame = old_frame[crop_parameters[0]:crop_parameters[1], crop_parameters[2]:crop_parameters[3]]
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    # Loop frames with frame skip constant
    success = True
    while success:
        cap.set(1, counter * frame_skip_constant)
        success, frame = cap.read()
        if success:
            # Process current frame
            frame = frame[crop_parameters[0]:crop_parameters[1], crop_parameters[2]:crop_parameters[3]]
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Calculate flow
            flow = cv2.calcOpticalFlowFarneback(old_gray, frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
            total_distance_list.append(np.mean(mag))

            # Now update the previous frame
            old_gray = frame_gray.copy()
        counter += 1
    cap.release()

    # Return statements
    print("Total execution time for dense optical flow:", vid_name, "-", (time.time() - start_time), "seconds")
    heat_map_module.numpy_io('write', location, np.array(total_distance_list))

# Get crop parameter dictionary

if exists("/media/nipek/My Book/Rabbit Research Videos/WP 3.2/Analysis/crop_parameters.json"):
    with open("/media/nipek/My Book/Rabbit Research Videos/WP 3.2/Analysis/crop_parameters.json", "r") as f:
        crop_parameters = json.load(f)
else:
    print('Create crop parameter dictionary!')

# Generate and save hourly heatmapsaction diagrams

argument_arr = []
for dir in glob.glob('/media/nipek/My Book/Rabbit Research Videos/WP 3.2/C*'):
    camera_text = dir.rsplit('/', 1)[1]
    if camera_text not in ['Camera 1']:
        action_diagram_path = os.path.join('/media/nipek/My Book/Rabbit Research Videos/WP 3.2/Analysis/Action_Diagrams', camera_text)

        # Create action diagram directories
        if not os.path.exists(action_diagram_path):
            os.mkdir(action_diagram_path)

        # Generate arguments
        for vid in sorted(glob.glob(dir + '/*')):
            vid_name = vid.rsplit('/', 1)[1].rsplit('.',1)[0]
            if not os.path.exists(os.path.join(action_diagram_path, (vid_name + '.npy'))):
                argument_arr.append((vid, crop_parameters[camera_text], vid_name,
                                     os.path.join(action_diagram_path, (vid_name + '.npy'))))

# Pool Processes
with mp.Pool(6) as p:
    p.starmap(calculate_action_dense_optical_flow, argument_arr)

## Checkpoint Complete!##