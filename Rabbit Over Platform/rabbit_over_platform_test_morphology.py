# Import statements

import os
from os.path import exists
import glob
import time
import sys
import json
import numpy as np
import heat_map_module
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as md
from scipy.signal import savgol_filter
import heat_map_module

# Test Function
def decrease_brightness(hsv_img, value=100):
    h, s, v = cv2.split(hsv_img)
    v[v > value] -= value
    v[v <= value] = 0
    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

def platform_usage_wp2_test(vid, crop_parameters, platform_parameters, minute, second, trs, mask):
    start_time = time.time()
    cap = cv2.VideoCapture(vid)

    ## Reading video frame by frame
    success = True
    while success:
        cap.set(1, 25*(minute*60+second))
        success, image = cap.read()
        if success:
            ### convert each image to grayscale
            gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)[crop_parameters[0]:crop_parameters[1], crop_parameters[2]:crop_parameters[3]][platform_parameters[0]:platform_parameters[1], :]
            frame_HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[crop_parameters[0]:crop_parameters[1], crop_parameters[2]:crop_parameters[3]][platform_parameters[0]:platform_parameters[1], :]
            cv2.imshow("HSV Image", frame_HSV)
            cv2.waitKey(0)
            gray_img[gray_img > 240] = 0
            dst = cv2.equalizeHist(gray_img)
            gray_img = cv2.cvtColor(decrease_brightness(frame_HSV), cv2.COLOR_BGR2GRAY)
            #dst = cv2.equalizeHist(gray_img)
            cv2.imshow("HSV Image", gray_img)
            cv2.waitKey(0)
            #thresh, bw_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            #thresh, bw_img = cv2.threshold(gray_img, thresh + trs, 255, cv2.THRESH_BINARY)
            #cv2.imshow("BW Image", bw_img)
            #cv2.waitKey(0)

            ### Fill contours less than 1000 pixels
            #kernel = np.ones((20, 5), np.uint8)
            #kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(20,2))
            #opening = cv2.morphologyEx(gray_img, cv2.MORPH_OPEN, kernel)
            #cv2.imshow("First Morphology Image", opening)
            #cv2.waitKey(0)
            result = gray_img.copy()

            low_threshold = 50
            high_threshold = 150
            edges = cv2.Canny(result, low_threshold, high_threshold)
            rho = 0.5
            # distance resolution in pixels of the Hough grid
            theta = np.pi / 180  # angular resolution in radians of the Hough grid
            threshold = 15  # minimum number of votes (intersections in Hough grid cell)
            min_line_length = 20  # minimum number of pixels making up a line
            max_line_gap = 5  # maximum gap in pixels between connectable line segments
            line_image = np.copy(result) * 0  # creating a blank to draw lines on

            # Run Hough on edge detected image
            # Output "lines" is an array containing endpoints of detected line segments
            lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)

            for line in lines:
                for x1, y1, x2, y2 in line:
                    cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            line_image
            line_image = cv2.bitwise_not(line_image)
            result = (line_image/255 * result).astype(np.uint8)
            thresh, result = cv2.threshold(result, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

            ### Find contours
            contours = cv2.findContours(result, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours = contours[0] if len(contours) == 2 else contours[1]
            cv2.imshow("XXX Image", result)
            cv2.waitKey(0)
            for c in contours:
                area = cv2.contourArea(c)
                if area < 100:
                    cv2.drawContours(result, [c], 0, color=(0, 0, 0), thickness=cv2.FILLED)

            for c in contours:
                area = cv2.contourArea(c)
                if area > 1000:
                    cv2.drawContours(result, [c], -1, color=(255, 255, 255), thickness=cv2.FILLED)


            #for c in contours:
            #    area = cv2.contourArea(c)
            #    if area > 2000:
            #        cv2.drawContours(result, [c], -1, color=(255, 255, 255), thickness=cv2.FILLED)
            cv2.imshow("Test Image", line_image)
            cv2.waitKey(0)
            cv2.imshow("Fill Contours Image", result)
            cv2.waitKey(0)
            ### Use 20x20 kernel for opening operation
            kernel = np.ones((15, 15), np.uint8)
            kernel2 = np.ones((1, 25), np.uint8)
            kernel3 = np.ones((25, 1), np.uint8)

            opening = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)
            cv2.imshow("Test Image", opening)
            cv2.waitKey(0)
            opening = cv2.morphologyEx(opening, cv2.MORPH_OPEN, kernel2)
            cv2.imshow("Test Image", opening)
            cv2.waitKey(0)
            opening = cv2.morphologyEx(opening, cv2.MORPH_OPEN, kernel3)
            cv2.imshow("Test Image", opening)
            cv2.waitKey(0)
            success = False


def platform_usage_wp2_test_2(vid, crop_parameters, platform_parameters, minute, second, threshold):
    start_time = time.time()
    cap = cv2.VideoCapture(vid)

    ## Reading video frame by frame
    success = True
    while success:
        cap.set(1, 25*(minute*60+second))
        success, image = cap.read()
        if success:
            ### convert each image to grayscale
            gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)[crop_parameters[0]:crop_parameters[1],
                       crop_parameters[2]:crop_parameters[3]][platform_parameters[0]:platform_parameters[1], :]
            gray_img[gray_img > 240] = 0
            thresh, bw_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            thresh, bw_img = cv2.threshold(gray_img, thresh + threshold, 255, cv2.THRESH_BINARY)

            ### Fill contours less than 1000 pixels
            result = bw_img.copy()
            kernel = np.ones((3, 3), np.uint8)
            opening = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)
            result = opening.copy()

            ### Find contours and fill
            contours = cv2.findContours(result, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours = contours[0] if len(contours) == 2 else contours[1]

            for c in contours:
                area = cv2.contourArea(c)
                if area > 500:
                    cv2.drawContours(result, [c], -1, color=(255, 255, 255), thickness=cv2.FILLED)

            for c in contours:
                area = cv2.contourArea(c)
                if area < 200:
                    cv2.drawContours(result, [c], -1, color=(0, 0, 0), thickness=cv2.FILLED)

            ### Use 20x20 kernel for opening operation
            kernel = np.ones((15, 15), np.uint8)
            kernel2 = np.ones((1, 25), np.uint8)
            kernel3 = np.ones((25, 1), np.uint8)

            opening = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)
            opening = cv2.morphologyEx(opening, cv2.MORPH_OPEN, kernel2)
            opening = cv2.morphologyEx(opening, cv2.MORPH_OPEN, kernel3)

            cv2.imshow("Test", opening)
            cv2.waitKey(0)
            success = False


def platform_usage_wp3_test_3(vid, crop_parameters, platform_parameters, minute, second, trs, mask):
    start_time = time.time()
    cap = cv2.VideoCapture(vid)

    ## Reading video frame by frame
    success = True
    while success:
        cap.set(1, 25*(minute*60+second))
        success, image = cap.read()
        if success:
            ### convert each image to grayscale
            frame_HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[crop_parameters[0]:crop_parameters[1], crop_parameters[2]:crop_parameters[3]][platform_parameters[0]:platform_parameters[1], :]
            gray_img = cv2.cvtColor(decrease_brightness(frame_HSV), cv2.COLOR_BGR2GRAY)
            result = gray_img.copy()
            cv2.imshow("Test Image", result)
            cv2.waitKey(0)

            ### Edge detection
            low_threshold = 50
            high_threshold = 150
            edges = cv2.Canny(result, low_threshold, high_threshold)

            ### Line enhancement
            rho = 0.5# distance resolution in pixels of the Hough grid
            theta = np.pi / 180  # angular resolution in radians of the Hough grid
            threshold = 10  # minimum number of votes (intersections in Hough grid cell)
            min_line_length = 10  # minimum number of pixels making up a line
            max_line_gap = 5  # maximum gap in pixels between connectable line segments
            line_image = np.copy(result) * 0  # creating a blank to draw lines on
            lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)
            for line in lines:
                for x1, y1, x2, y2 in line:
                    cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            line_image = cv2.bitwise_not(line_image)
            result = (line_image/255 * result).astype(np.uint8)

            ### Binary Threshold
            thresh, result = cv2.threshold(result, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            cv2.imshow("Test Image", result)
            cv2.waitKey(0)

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
            #kernel2 = np.ones((1, 25), np.uint8)
            #kernel3 = np.ones((25, 1), np.uint8)

            opening = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)
            cv2.imshow("Test Image", opening)
            cv2.waitKey(0)
            #opening = cv2.morphologyEx(opening, cv2.MORPH_OPEN, kernel2)
            #cv2.imshow("Test Image", opening)
            #cv2.waitKey(0)
            #opening = cv2.morphologyEx(opening, cv2.MORPH_OPEN, kernel3)
            #cv2.imshow("Test Image", opening)
            #cv2.waitKey(0)
            success = False
            print(np.count_nonzero(opening))
            return np.count_nonzero(opening)

if exists("/media/nipek/My Book/Rabbit Research Videos/WP 3.2/Analysis/crop_parameters.json"):
    with open("/media/nipek/My Book/Rabbit Research Videos/WP 3.2/Analysis/crop_parameters.json", "r") as f:
        crop_parameters = json.load(f)
else:
    print('Create crop parameter dictionary!')

if exists("/media/nipek/My Book/Rabbit Research Videos/WP 3.2/Analysis/Rabbit_Over_Platform/rabbit_over_platform_parameters.json"):
    with open("/media/nipek/My Book/Rabbit Research Videos/WP 3.2/Analysis/Rabbit_Over_Platform/rabbit_over_platform_parameters.json", "r") as f:
        rabbit_over_platform_parameters = json.load(f)
else:
    print('Create platform parameters dictionary!')


platform_usage_wp3_test_3('/media/nipek/My Book/Rabbit Research Videos/WP 3.2/Camera 12/kon12.20210701_020009.mp4',
                          crop_parameters['Camera 12'], rabbit_over_platform_parameters['Camera 12'],
                          10, 35, 40, '/media/nipek/My Book/Rabbit Research Videos/WP 3.2/Analysis/Rabbit_Over_Platform/Masks/Camera_5.jpg')


#platform_usage_wp2_test_3('/media/nipek/My Book/Rabbit Research Videos/WP 3.2/Camera 8/kon08.20210701_040000.mp4',
#                          crop_parameters['Camera 8'], rabbit_over_platform_parameters['Camera 8'],
#                          14, 47, 40)





#temp_arr = heat_map_module.numpy_io('read', '/media/nipek/My Book/Rabbit Research Videos/WP 3.2/Analysis/Rabbit_Over_Platform/Secondly_Sequences/Camera 25/kon25.20210630_220000.npy')
#print(np.count_nonzero(temp_arr))

def plot_upper_platform_usage_wp2(arr, apply_savgol_filter=False, filename=None):

    ## Time component
    time_arr = np.arange(0, arr.shape[0], 1)
    x_axis = pd.to_datetime(time_arr / 60, unit='m')
    xfmt = md.DateFormatter('%H:%M:%S')

    ## Y component and smoothing
    if apply_savgol_filter:
        y_axis = savgol_filter(arr, 11, 3)
    else:
        y_axis = arr

    ## Plot
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.plot(x_axis, y_axis, color='green')
    ax.set(xlabel='Time', ylabel='Detected White Pixel Count (After Morphology)', title='Upper Platform Usage')
    ax.grid()

    ### X-axis ticks
    ax.xaxis.set_major_formatter(xfmt)
    plt.xticks(rotation=90)
    minor_ticks = np.arange(0, arr.shape[0] + 1, 60)
    x_axis_minor = pd.to_datetime(minor_ticks / 60, unit='m')
    ax.set_xticks(x_axis_minor)

    if filename is not None:
        fig.savefig(filename)
    else:
        plt.show()
    ## Return void

#plot_upper_platform_usage_wp2(temp_arr, False, None)


### Redundant Code

# gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)[crop_parameters[0]:crop_parameters[1], crop_parameters[2]:crop_parameters[3]]
# thresh, bw_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
# thresh, bw_img = cv2.threshold(gray_img, thresh+40, 255, cv2.THRESH_BINARY)
# th = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11 , 5)
# bw_img = bw_img[platform_parameters[0]:platform_parameters[1], :]