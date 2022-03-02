# Import statements

import os
from os.path import exists
import glob
import time
import json
import numpy as np
import heat_map_module
import cv2
import pandas as pd
import matplotlib.dates as md
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
np.set_printoptions(threshold=5000)

# Functions

def spike_detect(arr):
    mean_spike = 0
    count = 0
    for element in range(len(arr)):
        if (arr[element] == 0) and ((element+1) < len(arr)) and (arr[(element+1)] > 0):
            mean_spike += arr[element+1]
            count += 1
    if count > 0:
        return mean_spike/count
    else:
        return 0

def min_detect(arr):
    return float(np.min(arr))

def get_spike(arr, window_size):
    arr -= np.min(arr)
    arr[arr < 500] = 0
    final_arr = arr.copy()
    for element in np.arange(window_size, len(arr)-window_size):
        sliced_arr = arr[element-window_size:element+window_size+1]
        total_zeroes = len(np.where(sliced_arr == 0)[0])
        if (total_zeroes/(2*window_size+1)) > 0.5:
            final_arr[element] = 0
    final_arr = np.array([np.mean(final_arr[i:i + 15]) for i in np.arange(0, len(final_arr) - 15 + 1)])
    final_arr_spike = final_arr.copy()
    final_arr_spike[final_arr_spike < np.quantile(final_arr_spike, 0.1)] = 0
    spike_height = spike_detect(final_arr_spike)
    return spike_height

# Mark Videos as Day & Night

video_dict = {
            "20210629_150000": "Day",
            "20210629_160000": "Day",
            "20210629_170000": "Day",
            "20210629_180000": "Day",
            "20210629_190000": "Day",
            "20210629_200000": "Day",
            "20210629_210000": "Day",
            "20210629_220000": "Day",
            "20210629_230000": "Day",
            "20210630_000000": "Mixed",
            "20210630_010000": "Night",
            "20210630_020009": "Night",
            "20210630_030000": "Night",
            "20210630_040000": "Night",
            "20210630_050000": "Night",
            "20210630_060000": "Night",
            "20210630_070000": "Mixed",
            "20210630_080000": "Day",
            "20210630_090000": "Day",
            "20210630_100000": "Day",
            "20210630_110000": "Day",
            "20210630_120000": "Day",
            "20210630_130000": "Day",
            "20210630_140000": "Day",
            "20210630_150000": "Day",
            "20210630_160000": "Day",
            "20210630_170000": "Day",
            "20210630_180000": "Day",
            "20210630_190000": "Day",
            "20210630_200000": "Mixed",
            "20210630_210000": "Night",
            "20210630_220000": "Night",
            "20210630_230000": "Night",
            "20210701_000000": "Night",
            "20210701_010000": "Night",
            "20210701_020009": "Night",
            "20210701_030000": "Night",
            "20210701_040000": "Night",
            "20210701_050000": "Night",
            "20210701_060000": "Night",
            "20210701_070000": "Night",
            "20210701_080000": "Mixed",
            "20210701_090000": "Day",
            "20210701_100000": "Day",
            "20210701_110000": "Day",
            "20210701_120000": "Day",
            "20210701_130000": "Day",
            "20210701_140000": "Day",
            "20210701_150000": "Day",
            "20210701_160000": "Day",
            "20210701_170000": "Day",
            "20210701_180000": "Day",
            "20210701_190000": "Day",
            "20210701_200000": "Mixed",
            "20210701_210000": "Night",
            "20210701_220000": "Night",
            "20210701_230000": "Night",
            "20210702_000000": "Night",
            "20210702_010000": "Night",
            "20210702_020009": "Night",
            "20210702_030000": "Night",
            "20210702_040000": "Night",
            "20210702_050000": "Night",
            "20210702_060000": "Night",
            "20210702_070000": "Night",
            "20210702_080000": "Mixed",
            "20210702_090000": "Day",
            "20210702_100000": "Day",
            "20210702_110000": "Day",
            "20210702_120000": "Day",
            "20210702_130000": "Day",
            "20210702_140000": "Day",
            "20210702_150000": "Day",
            "20210702_160000": "Day"}

if exists("/media/nipek/My Book/Rabbit Research Videos/WP 3.2/Analysis/Rabbit_Over_Platform/rabbit_over_platform_video_parameters.json"):
    with open("/media/nipek/My Book/Rabbit Research Videos/WP 3.2/Analysis/Rabbit_Over_Platform/rabbit_over_platform_video_parameters.json", "r") as f:
        video_parameters = json.load(f)
        print(video_parameters)
else:
    with open("/media/nipek/My Book/Rabbit Research Videos/WP 3.2/Analysis/Rabbit_Over_Platform/rabbit_over_platform_video_parameters.json", "w") as f:
        json.dump(video_dict, f, indent=4)
    with open("/media/nipek/My Book/Rabbit Research Videos/WP 3.2/Analysis/Rabbit_Over_Platform/rabbit_over_platform_video_parameters.json","r") as f:
            video_parameters = json.load(f)
            print(video_parameters)

day_sum = sum(value == 'Day' for value in video_parameters.values())
night_sum = sum(value == 'Night' for value in video_parameters.values())
mixed_sum = sum(value == 'Mixed' for value in video_parameters.values())
print('\n' ,'Total Day Light: ', day_sum, '\n',
            'Total Night Light: ', night_sum, '\n',
            'Total Mixed Light: ', mixed_sum)

# Create Spike Dict

spike_dict = {}

for dir in sorted(glob.glob('/media/nipek/My Book/Rabbit Research Videos/WP 3.2/C*')):
    camera_text = dir.rsplit('/', 1)[1]
    rabbit_over_platform_path = os.path.join('/media/nipek/My Book/Rabbit Research Videos/WP 3.2/Analysis/Rabbit_Over_Platform/Secondly_Sequences_HSV', camera_text)

    # Empty spike dict for camera
    temp_camera_spike = {}

    for vid in sorted(glob.glob(rabbit_over_platform_path + '/*.npy')):
        vid_name = vid.rsplit('/', 1)[1].rsplit('.', 1)[0].rsplit('.', 1)[1]
        if '020000' not in vid_name:
            temp_arr = heat_map_module.numpy_io('read', vid)
            min_temp_arr = min_detect(temp_arr)
            print(video_parameters[vid_name], min_temp_arr)
            # Update spike dict
            if video_parameters[vid_name] not in temp_camera_spike:
                temp_camera_spike[str(video_parameters[vid_name] + '_Min')] = min_temp_arr
                temp_camera_spike[video_parameters[vid_name]] = get_spike(temp_arr, 10)
            else:
                temp_camera_spike[str(video_parameters[vid_name] + '_Min')] += min_temp_arr
                temp_camera_spike[video_parameters[vid_name]] += get_spike(temp_arr, 10)

    # Normalize spike dict
    for time in ['Day', 'Night', 'Mixed', 'Day_Min', 'Night_Min', 'Mixed_Min']:
        if time == 'Day':
            temp_camera_spike[time] /= day_sum
        elif time == 'Night':
            temp_camera_spike[time] /= night_sum
        elif time == 'Mixed':
            temp_camera_spike[time] /= mixed_sum
        elif time == 'Day_Min':
            temp_camera_spike[time] /= day_sum
        elif time == 'Night_Min':
            temp_camera_spike[time] /= night_sum
        elif time == 'Mixed_Min':
            temp_camera_spike[time] /= mixed_sum

    # Add camera to spike dict
    spike_dict[camera_text] = temp_camera_spike

    # Create camera entry in spike dict

if exists("/media/nipek/My Book/Rabbit Research Videos/WP 3.2/Analysis/Rabbit_Over_Platform/rabbit_over_platform_spike_parameters.json"):
    with open("/media/nipek/My Book/Rabbit Research Videos/WP 3.2/Analysis/Rabbit_Over_Platform/rabbit_over_platform_spike_parameters.json", "r") as f:
        spike_parameters = json.load(f)
        print(spike_parameters)
else:
    with open("/media/nipek/My Book/Rabbit Research Videos/WP 3.2/Analysis/Rabbit_Over_Platform/rabbit_over_platform_spike_parameters.json", "w") as f:
        json.dump(spike_dict, f, indent=4)
    with open("/media/nipek/My Book/Rabbit Research Videos/WP 3.2/Analysis/Rabbit_Over_Platform/rabbit_over_platform_spike_parameters.json","r") as f:
            spike_parameters = json.load(f)
            print(spike_parameters)

