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


# Define platform usage function

def decrease_brightness(hsv_img, value=100):
    h, s, v = cv2.split(hsv_img)
    v[v > value] -= value
    v[v <= value] = 0
    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


def platform_usage_wp2_pool(vid, crop_parameters, platform_parameters, save_location):
    start_time = time.time()
    cap = cv2.VideoCapture(vid)

    ## Initialize global parameters
    counter = 0
    frame_skip_constant = 25

    ## Create empty Numpy array
    platform_use_frame = np.zeros(((int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / frame_skip_constant)) + 1),
                                  np.dtype('uint16'))

    ## Reading video frame by frame
    success = True
    while success:
        cap.set(1, counter * frame_skip_constant)
        success, image = cap.read()
        if success:
            ### convert each image to grayscale
            frame_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[crop_parameters[0]:crop_parameters[1],
                                                               crop_parameters[2]:crop_parameters[3]][platform_parameters[0]:platform_parameters[1], :]
            gray_img = cv2.cvtColor(decrease_brightness(frame_hsv), cv2.COLOR_BGR2GRAY)
            result = gray_img.copy()

            ### Edge detection
            low_threshold = 50
            high_threshold = 150
            edges = cv2.Canny(result, low_threshold, high_threshold)

            ### Line enhancement
            rho = 0.5  # distance resolution in pixels of the Hough grid
            theta = np.pi / 180  # angular resolution in radians of the Hough grid
            threshold = 10  # minimum number of votes (intersections in Hough grid cell)
            min_line_length = 10  # minimum number of pixels making up a line
            max_line_gap = 5  # maximum gap in pixels between connectable line segments
            line_image = np.copy(result) * 0  # creating a blank to draw lines on
            lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)
            if lines is not None:
                for line in lines:
                    for x1, y1, x2, y2 in line:
                        cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            line_image = cv2.bitwise_not(line_image)
            result = (line_image / 255 * result).astype(np.uint8)

            ### Binary Threshold
            thresh, result = cv2.threshold(result, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

            ### Find and fill contours
            contours = cv2.findContours(result, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours = contours[0] if len(contours) == 2 else contours[1]

            for c in contours:
                area = cv2.contourArea(c)
                if area < 500:
                    cv2.drawContours(result, [c], 0, color=(0, 0, 0), thickness=cv2.FILLED)
            for c in contours:
                area = cv2.contourArea(c)
                if area > 2000:
                    cv2.drawContours(result, [c], -1, color=(255, 255, 255), thickness=cv2.FILLED)

            ### Use 20x20 kernel for opening operation
            kernel = np.ones((10, 10), np.uint8)
            opening = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)

            ### Adjust the new created array
            platform_use_frame[counter] = np.count_nonzero(opening)
            counter += 1

    ## Return statements
    print("Total execution time for platform usage:", (time.time() - start_time), "seconds")
    heat_map_module.numpy_io('write', save_location, platform_use_frame)


# Get crop parameter dictionary

if exists("/media/nipek/My Book/Rabbit Research Videos/WP 3.2/Analysis/crop_parameters.json"):
    with open("/media/nipek/My Book/Rabbit Research Videos/WP 3.2/Analysis/crop_parameters.json", "r") as f:
        crop_parameters = json.load(f)
else:
    print('Create crop parameter dictionary!')

# Get platform parameters dictionary

if exists(
        "/media/nipek/My Book/Rabbit Research Videos/WP 3.2/Analysis/Rabbit_Over_Platform/rabbit_over_platform_parameters.json"):
    with open(
            "/media/nipek/My Book/Rabbit Research Videos/WP 3.2/Analysis/Rabbit_Over_Platform/rabbit_over_platform_parameters.json",
            "r") as f:
        rabbit_over_platform_parameters = json.load(f)
else:
    print('Create platform parameters dictionary!')

# Compute platform usage

for dir in sorted(glob.glob('/media/nipek/My Book/Rabbit Research Videos/WP 3.2/C*')):
    camera_text = dir.rsplit('/', 1)[1]
    if camera_text not in ['Camera 1']:
        rabbit_over_platform_path = os.path.join(
            '/media/nipek/My Book/Rabbit Research Videos/WP 3.2/Analysis/Rabbit_Over_Platform/Secondly_Sequences_HSV',
            camera_text)

        # Create rabbit over the platform directories
        if not os.path.exists(rabbit_over_platform_path):
            os.mkdir(rabbit_over_platform_path)

        # Generate arguments
        argument_arr = []
        for vid in sorted(glob.glob(dir + '/*')):
            vid_name = vid.rsplit('/', 1)[1].rsplit('.', 1)[0]
            argument_arr.append((vid, crop_parameters[camera_text], rabbit_over_platform_parameters[camera_text],
                                 os.path.join(rabbit_over_platform_path, (vid_name + '.npy'))))
        print(argument_arr)

        # Pool Processes
        with mp.Pool(6) as p:
            p.starmap(platform_usage_wp2_pool, argument_arr)

## Checkpoint Complete!##


# for vid in sorted(glob.glob(directory + '/*')):
#     vid_name = vid.rsplit('/', 1)[1].rsplit('.', 1)[0]
#     if not os.path.exists(os.path.join(rabbit_over_platform_path, (vid_name + '.npy'))):
#         platform_usage_wp2_pool(vid, crop_parameters[camera_text], rabbit_over_platform_parameters[camera_text],
#                                 os.path.join(rabbit_over_platform_path, (vid_name + '.npy')))
