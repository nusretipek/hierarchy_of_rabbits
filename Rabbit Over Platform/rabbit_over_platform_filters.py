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

# Eliminate noise around 0 values

def window_noise_removal(arr, window_size):
    #arr[arr < np.quantile(arr, 0.25)] = np.quantile(arr, 0.25)
    print(np.min(arr))
    arr -= 2200
    arr[arr < 500] = 0
    #arr[np.where(arr >= 2200)] -= 2200

    for element in np.arange(window_size, len(arr)-window_size):
        sliced_arr = arr[element-window_size:element+window_size+1]
        total_zeroes = len(np.where(sliced_arr == 0)[0])
        if (total_zeroes/(2*window_size+1)) > 0.5:
            arr[element] = 0

    #arr[np.where(arr < np.quantile(arr, 0.9)- np.quantile(arr, 0.1))] = 0
    #arr[arr < 2700] = 0
    return arr

def spike_detect(arr, forward_window):
    mean_spike = 0
    count = 0
    for element in range(len(arr)-forward_window):
        if arr[element] == 0:
            slice = arr[element+1:element+forward_window]
            print(len(np.where(slice == 0)[0]))
            if  (14 > len(np.where(slice == 0)[0]) > 0):
                print(element)
                mean_spike += np.max(slice)
                count += 1
    return mean_spike/1

def spike_detect_2(arr):
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

# Filter spikes and combine the drop spikes
def remove_noisy_spikes(arr, window, window_2):
    final_arr = arr.copy()
    for element in range(len(arr)):
        if (arr[element] != 0) and ((element+window) < len(arr)) and ((element-window) > 0):
            slice_pre = arr[element-window:element]
            slice_post = arr[element:element+window+1]
            ratio_of_zeros = (np.count_nonzero(slice_pre) + np.count_nonzero(slice_post))/(window*2+1)
            if (ratio_of_zeros < 0.9):
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

def window_noise_removal_2(arr, window_size):
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
    spike_height = spike_detect_2(final_arr_spike)
    print(spike_height)
    #final_arr -= spike_height
    final_arr[final_arr < 1000] = 0

    return final_arr

# Assign median value

def get_index_slices(arr, number_of_zeroes):
    final_arr = np.repeat(0, arr.shape[0])
    zeros_np = np.repeat(0, number_of_zeroes)
    index_list = [i for i in range(0, len(arr)-number_of_zeroes) if list(arr[i:i+number_of_zeroes])!=list(zeros_np)]
    index_diff = np.diff(np.array(index_list))
    start_point = index_list[0]
    for element in np.where(index_diff > 10)[0]:
        end_point = index_list[element]
        final_arr[start_point:end_point] = np.repeat(np.median(arr[start_point:index_list[element]]), end_point-start_point)
        start_point = index_list[element+1]
    if 3599-number_of_zeroes in index_list:
        final_arr[start_point:3599] = np.repeat(np.median(arr[start_point:3599]), 3599-start_point)
    else:
        final_arr[start_point:index_list[-1]] = np.repeat(np.median(arr[start_point:index_list[-1]]), index_list[-1]-start_point)
    return final_arr

def get_index_slices_2(arr, number_of_zeroes):
    final_arr = np.repeat(0, arr.shape[0])
    zeros_np = np.repeat(0, number_of_zeroes)
    index_list = [i for i in range(0, len(arr)-number_of_zeroes) if list(arr[i:i+number_of_zeroes])!=list(zeros_np)]
    index_diff = np.diff(np.array(index_list))
    start_point = index_list[0]
    for element in np.where(index_diff > 10)[0]:
        end_point = index_list[element]
        final_arr[start_point:end_point] = np.repeat(np.median(arr[start_point:index_list[element]]), end_point-start_point)
        start_point = index_list[element+1]
    if 3599-number_of_zeroes in index_list:
        final_arr[start_point:3599] = np.repeat(np.median(arr[start_point:3599]), 3599-start_point)
    else:
        final_arr[start_point:index_list[-1]] = np.repeat(np.median(arr[start_point:index_list[-1]]), index_list[-1]-start_point)
    #print(np.diff(np.where(final_arr > 0)[0]))
    last_peak = None
    for element in range(len(final_arr)):
        if (final_arr[element] > 0 and last_peak is None):
            last_peak = element
        if (final_arr[element] > 0):
            if (1 < (element - last_peak) < 180):
                median = 0.5*(final_arr[element] + final_arr[last_peak])
                final_arr[last_peak:element] = np.repeat(median, element-last_peak)
            last_peak = element
            last_median = final_arr[element]
    return final_arr

## Checkpoint Complete!##

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

temp_arr = heat_map_module.numpy_io('read', '/media/nipek/My Book/Rabbit Research Videos/WP 3.2/Analysis/Rabbit_Over_Platform/Secondly_Sequences_HSV/Camera 8/kon08.20210701_150000.npy')

plot_upper_platform_usage_wp2(temp_arr)
temp_arr = window_noise_removal_2(temp_arr, 10)
plot_upper_platform_usage_wp2(temp_arr, apply_savgol_filter=False, filename=None)
temp_arr = remove_noisy_spikes(temp_arr, 30, 10)
temp_arr = combine_drop_spikes(temp_arr, 60)
temp_arr = remove_noisy_spikes(temp_arr, 10, 5)
temp_arr = combine_drop_spikes(temp_arr, 30)
plot_upper_platform_usage_wp2(temp_arr)



#plot_upper_platform_usage_wp2(get_index_slices(temp_arr, 10), apply_savgol_filter=False, filename=None)
#plot_upper_platform_usage_wp2(get_index_slices_2(window_noise_removal_2(temp_arr, 10), 10), apply_savgol_filter=False, filename=None)

#temp_arr = window_noise_removal(temp_arr, 20)
#print(temp_arr)