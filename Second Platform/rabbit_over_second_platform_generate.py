# Import statements

import os
from os.path import exists
import glob
import time
import json
import numpy as np
import heat_map_module
import cv2
import multiprocessing as mp
import sys

# Define platform usage function

def decrease_brightness(hsv_img, value=100):
    h, s, v = cv2.split(hsv_img)
    v[v > value] -= value
    v[v <= value] = 0
    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

def second_platform_usage_wp1_pool(vid, crop_parameters, save_location):
    start_time = time.time()
    cap = cv2.VideoCapture(vid)

    ## Initialize global parameters
    counter = 0
    frame_skip_constant = 25

    ## Create empty Numpy array
    platform_use_frame = np.zeros(((int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / frame_skip_constant))+1), np.dtype('uint16'))

    ## Reading video frame by frame
    success = True
    while success:
        cap.set(1, counter * frame_skip_constant)
        success, image = cap.read()
        if success:

            ### convert each image to grayscale
            frame_HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[crop_parameters[0]:crop_parameters[1],crop_parameters[2]:crop_parameters[3]]
            gray_img = cv2.cvtColor(decrease_brightness(frame_HSV, 80), cv2.COLOR_BGR2GRAY)
            result = gray_img.copy()

            ### Use 3x3 kernel for opening operation
            kernel = np.ones((2, 2), np.uint8)
            result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)
            kernel = np.ones((10, 10), np.uint8)
            closing = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
            kernel = np.ones((5, 5), np.uint8)
            opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)

            ### Adjust the new created array
            platform_use_frame[counter] = np.count_nonzero(opening)
            counter += 1
            if counter == len(platform_use_frame):
                success = False

    ## Return statements
    print("Total execution time for platform usage:", (time.time() - start_time), "seconds")
    heat_map_module.numpy_io('write', save_location, platform_use_frame)


# Get crop parameter dictionary

if exists("/media/nipek/My Book/Rabbit Research Videos/WP 3.1 - ILVO/Analysis/Second_Platform/crop_parameters.json"):
    with open("/media/nipek/My Book/Rabbit Research Videos/WP 3.1 - ILVO/Analysis/Second_Platform/crop_parameters.json", "r") as f:
        crop_parameters = json.load(f)
else:
    print('Create crop parameter dictionary!')

#Cycle #1

# Create rabbit over the platform directories
rabbit_over_platform_path = '/media/nipek/My Book/Rabbit Research Videos/WP 3.1 - ILVO/Analysis/Second_Platform/Cycle 1/'
for camera_no in crop_parameters:
    if not os.path.exists(os.path.join(rabbit_over_platform_path, ('Camera ' + camera_no))):
        os.mkdir(os.path.join(rabbit_over_platform_path, ('Camera ' + camera_no)))

# Compute platform usage

argument_arr = []
for vid in sorted(glob.glob('/media/nipek/My Book/Rabbit Research Videos/WP 3.1 - ILVO/Cycle 1/*')):
    camera_number = vid.rsplit('.', 2)[0].rsplit('/', 1)[1][3:]
    if (camera_number in crop_parameters):

        # Generate arguments
        vid_name = vid.rsplit('/', 1)[1].rsplit('.',1)[0]
        argument_arr.append((vid, crop_parameters[camera_number][0],
                             os.path.join(rabbit_over_platform_path, ('Camera ' + camera_no), (vid_name + '_0.npy'))))
        argument_arr.append((vid, crop_parameters[camera_number][1],
                             os.path.join(rabbit_over_platform_path, ('Camera ' + camera_no), (vid_name + '_1.npy'))))

#Cycle #2

# Create rabbit over the platform directories
rabbit_over_platform_path = '/media/nipek/My Book/Rabbit Research Videos/WP 3.1 - ILVO/Analysis/Second_Platform/Cycle 2/'
for camera_no in crop_parameters:
    if not os.path.exists(os.path.join(rabbit_over_platform_path, ('Camera ' + camera_no))):
        os.mkdir(os.path.join(rabbit_over_platform_path, ('Camera ' + camera_no)))

# Compute platform usage

for vid in sorted(glob.glob('/media/nipek/My Book/Rabbit Research Videos/WP 3.1 - ILVO/Cycle 2/*')):
    camera_number = vid.rsplit('.', 2)[0].rsplit('/', 1)[1][3:]
    if (camera_number in crop_parameters):

        # Generate arguments
        vid_name = vid.rsplit('/', 1)[1].rsplit('.',1)[0]
        argument_arr.append((vid, crop_parameters[camera_number][0],
                             os.path.join(rabbit_over_platform_path, ('Camera ' + camera_no), (vid_name + '_0.npy'))))
        argument_arr.append((vid, crop_parameters[camera_number][1],
                             os.path.join(rabbit_over_platform_path, ('Camera ' + camera_no), (vid_name + '_1.npy'))))

with mp.Pool(6) as p:
    p.starmap(second_platform_usage_wp1_pool, argument_arr)

## Checkpoint Complete!##