# Import statements

import numpy as np
import pandas as pd
import matplotlib.dates as md
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
np.set_printoptions(threshold=5000)

# Functions

def window_noise_removal(arr, window_size):
    arr -= np.min(arr)
    arr[arr < 1] = 0
    final_arr = arr.copy()
    for element in np.arange(window_size, len(arr)-window_size):
        sliced_arr = arr[element-window_size:element+window_size+1]
        total_zeroes = len(np.where(sliced_arr == 0)[0])
        if (total_zeroes/(2*window_size+1)) > 0.5:
            final_arr[element] = 0
    final_arr = np.array([np.mean(final_arr[i:i + 15]) for i in np.arange(0, len(final_arr) - 15 + 1)])
    final_arr[final_arr < 0.5] = 0
    return final_arr

def expand_intervals(arr, expansion_size, operation):
    final_arr = arr.copy()
    next_non_zero = 0
    prev_index = 0
    for index in range(expansion_size,len(arr) - expansion_size):
        if (prev_index+next_non_zero+expansion_size > index):
            continue
        if (arr[index] > 0):
            next_non_zero = 0
            while((index+next_non_zero) < (len(arr) - expansion_size) and arr[index + next_non_zero] != 0):
                next_non_zero += 1
            if operation == 'max':
                slice_value = np.max(arr[index:index+next_non_zero])
            if operation == 'mean':
                slice_value = np.mean(arr[index:index+next_non_zero])
            if (index+next_non_zero+expansion_size < len(final_arr)):
                final_arr[index-expansion_size:index+next_non_zero+expansion_size] = np.repeat(slice_value,2*expansion_size+next_non_zero)
            else:
                final_arr[index-expansion_size:len(arr)] = np.repeat(slice_value, len(arr)-(index-expansion_size))
            prev_index = index
    return final_arr

def combine_close_actions(arr, window):
    final_arr = arr.copy()
    non_zero_indices = np.where(arr > 0)
    non_zero_diff = np.diff(non_zero_indices)
    for element in np.where(np.logical_and(1 < non_zero_diff[0], non_zero_diff[0] < window))[0]:
        slice_max = np.max(arr[non_zero_indices[0][element]:non_zero_indices[0][element+1]])
        final_arr[non_zero_indices[0][element]:non_zero_indices[0][element+1]] = np.repeat(slice_max, non_zero_indices[0][element+1]-non_zero_indices[0][element])
    return final_arr

def get_index_slices(arr):
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

def combine_drop_spikes(arr, window):
    arr = final_arr.copy()
    next_non_zero = 0
    prev_index = 0
    for index in range(0,len(arr)):
        if (prev_index+next_non_zero > index):
            continue
        if (arr[index] > 0):
            while((index + next_non_zero) < len(arr) and arr[index + next_non_zero] != 0):
                next_non_zero += 1
            slice_max = np.max(arr[index:index+next_non_zero])
            if (index+next_non_zero < len(final_arr)):
                final_arr[index:index+next_non_zero] = np.repeat(slice_max,next_non_zero)
            else:
                final_arr[index:len(arr)] = np.repeat(slice_max, len(arr)-(index))
            prev_index = index
        next_non_zero = 0
    return final_arr

def plot_action_diagrams_wp2(arr, apply_savgol_filter=False, filename=None):

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
    ax.set(xlabel='Time', ylabel='Activity Mean (Dense Optical Flow)', title='Action Diagram')
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
    plt.close()
    ## Return void

## Checkpoint Complete!##