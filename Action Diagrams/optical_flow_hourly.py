# Import statements

import os
from os.path import exists
import glob
import numpy as np
import pandas as pd
import json
import heat_map_module
import matplotlib.pyplot as plt
import matplotlib.dates as md
from scipy.signal import savgol_filter

# Get crop parameters dictionary

if exists("/media/nipek/My Book/Rabbit Research Videos/WP 3.2/Analysis/cage_open_dict.json"):
    with open("/media/nipek/My Book/Rabbit Research Videos/WP 3.2/Analysis/cage_open_dict.json", "r") as f:
        cage_open_dict = json.load(f)
else:
    print('Create cage-open moments dictionary! Else, the feeding series will ignore it')

# Define plot function

def plot_optical_flow_hourly_wp2(arr, time_arr, camera_text, cage_open_dict, apply_savgol_filter = False, filename = None):

    ## Time component
    x_axis = time_arr[0]
    xfmt = md.DateFormatter('%d/%m %H:%M')

    ## Y component and smoothing
    if apply_savgol_filter:
        y_axis = savgol_filter(arr, 11, 3)
    else:
        y_axis = arr

    ## Plot
    fig, ax = plt.subplots(figsize = (20, 8))
    fig.subplots_adjust(bottom = 0.2)
    ax.plot(x_axis, y_axis, "-o", color = 'steelblue', markerfacecolor = 'steelblue', markersize = 7, label = 'Action Seconds')
    ax.plot(x_axis, np.zeros(arr.shape[0]) + np.mean(arr), color = 'limegreen', label = 'Mean Action Seconds') #Plot mean line

    ## Add trendline
    z = np.polyfit(np.arange(y_axis.shape[0], dtype = 'float64'), y_axis.astype('float64'), 1)
    p = np.poly1d(z)
    ax.plot(x_axis, p(np.arange(y_axis.shape[0],dtype = 'float64')), "-", color = 'purple', label = 'Trend Line')

    ## Adjust markers and add legend (cage-open)
    cage_open_arr = cage_open_dict[camera_text]
    cage_open_x = []
    cage_open_y = []
    for time_point in cage_open_arr:
        date_time_object = pd.to_datetime(time_point, format='%Y%m%d%H%M%S')
        index = np.where(time_arr[0] == date_time_object)
        cage_open_x.append(time_arr[0][index])
        cage_open_y.append(y_axis[index])
    ax.plot(cage_open_x, cage_open_y, "s", color = 'crimson', markersize = 10, label = 'Open Cage')
    plt.legend(prop={'size': 8})

    ## Edit plot title and labels
    ax.set(xlabel = 'Time', ylabel = 'Count of Seconds', title = 'Action Count of Seconds Chart' + ' (' + camera_text + ')')
    ax.grid()

    ## X-axis ticks
    ax.xaxis.set_major_formatter(xfmt)
    plt.xticks(rotation = 90)
    x_axis_minor = time_arr[0]
    ax.set_xticks(x_axis_minor)

    ## Save figure
    if filename is not None:
        fig.savefig(filename)
    else:
        plt.show()
    plt.close()

# Hourly aggregate filtered arrays
for dir in sorted(glob.glob('/media/nipek/My Book/Rabbit Research Videos/WP 3.2/C*')):
    camera_text = dir.rsplit('/', 1)[1]
    action_analysis_path = os.path.join('/media/nipek/My Book/Rabbit Research Videos/WP 3.2/Analysis/Action_Diagrams', camera_text)
    time_arr_location = '/media/nipek/My Book/Rabbit Research Videos/WP 3.2/Analysis/Feeding_Time_Series/' + camera_text + '.npy'

    # Load time numpy array
    with open(time_arr_location, 'rb') as f:
        temp_time_arr = np.load(f, allow_pickle = True)

    hourly_action = []
    # Generate aggregated (hourly) platform usage arrays 1D numpy
    for action_arr in sorted(glob.glob(action_analysis_path + '/*filtered.npy')):
        if '020000' not in action_arr:
            vid_text = action_arr.rsplit('/', 1)[1].rsplit('.', 1)[0]
            temp_arr = heat_map_module.numpy_io('read', action_arr)
            hourly_action.append(np.count_nonzero(temp_arr))

    # Plot and save
    save_location_reg = '/media/nipek/My Book/Rabbit Research Videos/WP 3.2/Analysis/Action_Diagrams/Hourly_Sequences/' + camera_text + '_plot.jpg'
    save_location_savgol = '/media/nipek/My Book/Rabbit Research Videos/WP 3.2/Analysis/Action_Diagrams/Hourly_Sequences/' + camera_text + '_savgol_plot.jpg'
    plot_optical_flow_hourly_wp2(np.array(hourly_action), temp_time_arr,  camera_text,
                                  cage_open_dict, apply_savgol_filter = False, filename = save_location_reg)
    plot_optical_flow_hourly_wp2(np.array(hourly_action), temp_time_arr,  camera_text,
                                  cage_open_dict, apply_savgol_filter = True, filename = save_location_savgol)

## Checkpoint Complete!##