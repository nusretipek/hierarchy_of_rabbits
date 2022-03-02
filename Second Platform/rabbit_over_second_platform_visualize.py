# Import statements

import os
from os.path import exists
import glob
import json
import numpy as np
import heat_map_module
import pandas as pd
import matplotlib.dates as md
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
np.set_printoptions(threshold=5000)
import sys

# Plot function for upper platform usage

def window_noise_removal(arr, window_size):
    arr -= np.min(arr)
    arr[arr < np.quantile(arr, 0.01)] = np.quantile(arr, 0.01)
    arr[arr > np.quantile(arr, 0.99)] = np.quantile(arr, 0.99)
    arr[np.mean(arr) > arr] = 0
    final_arr = arr.copy()
    for element in np.arange(window_size, len(arr)-window_size):
        sliced_arr = arr[element-window_size:element+window_size+1]
        total_zeroes = len(np.where(sliced_arr == 0)[0])
        if (total_zeroes/(2*window_size+1)) > 0.8:
            final_arr[element] = 0
    final_arr = np.array([np.mean(final_arr[i- 3:i + 3]) for i in np.arange(3, len(final_arr) - 3 + 1)])
    final_arr[final_arr < 3000] = 0
    return final_arr

def remove_noisy_spikes(arr, window, window_2):
    final_arr = arr.copy()
    for element in range(len(arr)):
        if (arr[element] != 0) and ((element+window) < len(arr)) and ((element-window) > 0):
            slice_pre = arr[element-window:element]
            slice_post = arr[element:element+window+1]
            ratio_of_zeros = (np.count_nonzero(slice_pre) + np.count_nonzero(slice_post))/(window*2+1)
            if (ratio_of_zeros < 0.5):
                final_arr[element] = 0
        if (arr[element] != 0) and arr[element-1] == 0 and ((element+window_2) < len(arr)) and ((element-1) > 0):
            bool_state =False
            for i in range(window_2):
                if (arr[element+i] == 0):
                    bool_state = True
            if bool_state:
                for i in range(window_2):
                    final_arr[element+i] = 0
    return final_arr

def combine_drop_spikes(arr, window):
    final_arr = arr.copy()
    for element in range(len(arr)):
        if (arr[element] == 0) and ((element+window) < len(arr)) and ((element-window) > 0):
            slice_pre = arr[element-window:element]
            slice_post = arr[element:element+window+1]
            if (np.count_nonzero(slice_pre) > (window/10) and np.count_nonzero(slice_post) > (window/10)):
                final_arr[element] = 0.5 * (np.max(slice_pre) + np.max(slice_post))
    return final_arr

def plot_upper_platform_usage_wp1(arr, apply_savgol_filter=False, filename=None):

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
    ax.set(xlabel='Time', ylabel='Detected White Pixel Count (After Morphology)', title='Second Upper Platform Usage')
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

temp_arr = heat_map_module.numpy_io('read', '/media/nipek/My Book/Rabbit Research Videos/WP 3.1 - ILVO/Analysis/Second_Platform/kon01_1.npy')
temp_arr = window_noise_removal(temp_arr, 10)
temp_arr = remove_noisy_spikes(temp_arr, 30, 10)
temp_arr = combine_drop_spikes(temp_arr, 60)
plot_upper_platform_usage_wp1(temp_arr)
sys.exit(0)

# Compute & visualize the platform usage (Per Hour)

## Load JSON files

if exists("/media/nipek/My Book/Rabbit Research Videos/WP 3.2/Analysis/Rabbit_Over_Platform/rabbit_over_platform_video_parameters.json"):
    with open("/media/nipek/My Book/Rabbit Research Videos/WP 3.2/Analysis/Rabbit_Over_Platform/rabbit_over_platform_video_parameters.json", "r") as f:
        video_parameters = json.load(f)
else:
    print("Create video parameters dictionary!")

if exists("/media/nipek/My Book/Rabbit Research Videos/WP 3.2/Analysis/Rabbit_Over_Platform/rabbit_over_platform_spike_parameters.json"):
    with open("/media/nipek/My Book/Rabbit Research Videos/WP 3.2/Analysis/Rabbit_Over_Platform/rabbit_over_platform_spike_parameters.json", "r") as f:
        spike_parameters = json.load(f)
else:
    print("Create spike parameters dictionary!")

if exists("/media/nipek/My Book/Rabbit Research Videos/WP 3.2/Analysis/Rabbit_Over_Platform/rabbit_over_platform_brightness_parameters.json"):
    with open("/media/nipek/My Book/Rabbit Research Videos/WP 3.2/Analysis/Rabbit_Over_Platform/rabbit_over_platform_brightness_parameters.json", "r") as f:
        brightness_parameters = json.load(f)
else:
    print("Create spike parameters dictionary!")


for dir in sorted(glob.glob('/media/nipek/My Book/Rabbit Research Videos/WP 3.2/C*')):
    camera_text = dir.rsplit('/', 1)[1]
    rabbit_over_platform_path = os.path.join('/media/nipek/My Book/Rabbit Research Videos/WP 3.2/Analysis/Rabbit_Over_Platform/Secondly_Sequences_HSV', camera_text)

    # Create rabbit over the platform directories
    if not os.path.exists(rabbit_over_platform_path):
        os.mkdir(rabbit_over_platform_path)

    # Generate platform usage filtered arrays 1D numpy
    for platform_usage_arr in sorted(glob.glob(rabbit_over_platform_path + '/*.npy')):
        if '020000' not in platform_usage_arr:
            vid_text = platform_usage_arr.rsplit('/', 1)[1].rsplit('.', 1)[0]
            video_time = vid_text.rsplit('.', 1)[1]
            light_parameter = video_parameters[video_time]
            temp_arr = heat_map_module.numpy_io('read', platform_usage_arr)
            temp_arr = window_noise_removal(temp_arr, 10, spike_parameters[camera_text][light_parameter + '_Min'], brightness_parameters[camera_text])
            temp_arr = remove_noisy_spikes(temp_arr, 30, 10)
            temp_arr = combine_drop_spikes(temp_arr, 60)
            temp_arr = remove_noisy_spikes(temp_arr, 10, 5)
            temp_arr = combine_drop_spikes(temp_arr, 30)
            #plot_upper_platform_usage_wp2(temp_arr, apply_savgol_filter=False,
            #                              filename=rabbit_over_platform_path + '/' + vid_text + '_filtered.png')
            heat_map_module.numpy_io('write', os.path.join(rabbit_over_platform_path, (vid_text + '_filtered.npy')), temp_arr)

## Checkpoint Complete!##