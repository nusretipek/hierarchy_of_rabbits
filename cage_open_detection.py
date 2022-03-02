# Import statements

import cv2
import heat_map_module
import time
import numpy as np
import glob
import pandas as pd
import json

# Morphological test for open cage

def test_cage_open_wp2(heatmap_file):
    img = cv2.imread(heatmap_file, cv2.IMREAD_GRAYSCALE)
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 7, 1)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for i in contours:
        area = cv2.contourArea(i)
        if area > 1:
            thresh = cv2.drawContours(thresh, i, 0, 255, thickness=cv2.FILLED)
    img2 = np.zeros((img.shape[0], img.shape[1]))
    cv2.fillPoly(img2, pts=contours, color=(255, 255, 255))
    closing = cv2.morphologyEx(img2, cv2.MORPH_CLOSE, np.ones((1,5),np.uint8))
    return heatmap_file.rsplit('/', 1)[1].rsplit('.', 1)[0], np.sum(closing == 255)

def detect_outliers_wp2(data):
    index_list = []
    m = np.median(data)
    s = np.std(data)
    for index in range(len(data)):
        if not (data[index] < m + 1.5 * s):
            index_list.append(index)
    return index_list

# Detect open cages

cage_open_dict = {}
for dir in sorted(glob.glob('/media/nipek/My Book/Rabbit Research Videos/WP 3.2/Analysis/Heatmaps/C*')):
    camera_text = dir.rsplit('/', 1)[1]
    white_pixel_list = []
    camera_name_list = []
    for heat_map in sorted(glob.glob(dir + '/*.jpg')):
        if ('020000' not in heat_map):
            date_time_text = heat_map.rsplit('/', 1)[1].rsplit('.', 1)[0].rsplit('.', 1)[1].replace('_', '').replace('heatmap', '')
            date_time_object = pd.to_datetime(date_time_text, format='%Y%m%d%H%M%S')
            name, white_pixel = test_cage_open_wp2(heat_map)
            camera_name_list.append(date_time_text)
            white_pixel_list.append(white_pixel)

    index_list = detect_outliers_wp2(np.array(white_pixel_list))
    cage_open_dict[camera_text] = np.array(camera_name_list)[index_list].tolist()

# Create open cage dictionary

with open("/media/nipek/My Book/Rabbit Research Videos/WP 3.2/Analysis/cage_open_dict.json", "w") as f:
    json.dump(cage_open_dict, f, indent = 4)

## Checkpoint Complete!##
