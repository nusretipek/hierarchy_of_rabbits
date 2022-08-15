# Import statements

import os
from os.path import exists
import glob
import time
import json
import numpy as np
import heat_map_module
import cv2

# Define platform usage function


def platform_usage_wp2(vid, crop_parameters, platform_parameters):
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
            gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)[crop_parameters[0]:crop_parameters[1], crop_parameters[2]:crop_parameters[3]]
            thresh, bw_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            thresh, bw_img = cv2.threshold(gray_img, thresh + 40, 255, cv2.THRESH_BINARY)
            bw_img = bw_img[platform_parameters[0]:platform_parameters[1], :]

            ### Find contours
            contours = cv2.findContours(bw_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours = contours[0] if len(contours) == 2 else contours[1]

            ### Fill contours less than 1000 pixels
            result = bw_img.copy()
            for c in contours:
                area = cv2.contourArea(c)
                if area < 500:
                    cv2.drawContours(result, [c], -1, color=(0, 0, 0), thickness=cv2.FILLED)

            ### Use 20x20 kernel for opening operation
            kernel = np.ones((20, 20), np.uint8)
            kernel2 = np.ones((1, 25), np.uint8)
            opening = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)
            opening = cv2.morphologyEx(opening, cv2.MORPH_OPEN, kernel2)

            ### Adjust the new created array
            platform_use_frame[counter] = np.count_nonzero(opening)
            counter += 1

    ## Return statements
    print("Total execution time for platform usage:", (time.time() - start_time), "seconds")
    return platform_use_frame


# Get crop parameter dictionary

if exists("/media/nipek/My Book/Rabbit Research Videos/WP 3.2/Analysis/crop_parameters.json"):
    with open("/media/nipek/My Book/Rabbit Research Videos/WP 3.2/Analysis/crop_parameters.json", "r") as f:
        crop_parameters = json.load(f)
else:
    print('Create crop parameter dictionary!')

# Get platform parameters dictionary

if exists("/media/nipek/My Book/Rabbit Research Videos/WP 3.2/Analysis/Rabbit_Over_Platform/rabbit_over_platform_parameters.json"):
    with open("/media/nipek/My Book/Rabbit Research Videos/WP 3.2/Analysis/Rabbit_Over_Platform/rabbit_over_platform_parameters.json", "r") as f:
        rabbit_over_platform_parameters = json.load(f)
else:
    print('Create platform parameters dictionary!')

# Compute platform usage

for dir in sorted(glob.glob('/media/nipek/My Book/Rabbit Research Videos/WP 3.2/C*')):
    camera_text = dir.rsplit('/', 1)[1]
    rabbit_over_platform_path = os.path.join('/media/nipek/My Book/Rabbit Research Videos/WP 3.2/Analysis/Rabbit_Over_Platform/Secondly_Sequences', camera_text)

    # Create rabbit over the platform directories
    if not os.path.exists(rabbit_over_platform_path):
        os.mkdir(rabbit_over_platform_path)

    # Generate platform usage arrays 1D numpy
    for vid in sorted(glob.glob(dir + '/*')):
        vid_name = vid.rsplit('/', 1)[1].rsplit('.', 1)[0]
        temp_arr = platform_usage_wp2(vid, crop_parameters[camera_text], rabbit_over_platform_parameters[camera_text])
        heat_map_module.numpy_io('write', os.path.join(rabbit_over_platform_path, (vid_name + '.npy')), temp_arr)

## Checkpoint Complete!##
