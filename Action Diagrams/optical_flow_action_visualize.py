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
import optical_flow_filters
from scipy.signal import savgol_filter

# Get crop parameters dictionary

if exists("/media/nipek/My Book/Rabbit Research Videos/WP 3.2/Analysis/cage_open_dict.json"):
    with open("/media/nipek/My Book/Rabbit Research Videos/WP 3.2/Analysis/cage_open_dict.json", "r") as f:
        cage_open_dict = json.load(f)
else:
    print('Create cage-open moments dictionary! Else, the feeding series will ignore it')

# Define plot function

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

for dir in sorted(glob.glob('/media/nipek/My Book/Rabbit Research Videos/WP 3.2/C*')):
    camera_text = dir.rsplit('/', 1)[1]
    action_diagram_path = os.path.join('/media/nipek/My Book/Rabbit Research Videos/WP 3.2/Analysis/Action_Diagrams', camera_text)

    # Create action diagram directories
    if not os.path.exists(action_diagram_path):
        os.mkdir(action_diagram_path)

    # Generate action diagrams from 1D numpy arrays
    for action_arr in sorted(glob.glob(action_diagram_path + '/*.npy')):
        if '020000' not in action_arr:
            vid_text = action_arr.rsplit('/', 1)[1].rsplit('.', 1)[0]
            temp_arr = heat_map_module.numpy_io('read', action_arr)

            # Define locations
            save_loc_nosav = action_diagram_path + '/' + vid_text +'.png'
            save_loc_sav = action_diagram_path + '/' + vid_text + '_savgol.png'
            save_loc_filtered = action_diagram_path + '/' + vid_text + '_filtered.png'

            # Plot
            plot_action_diagrams_wp2(temp_arr, apply_savgol_filter = False, filename = save_loc_nosav)
            plot_action_diagrams_wp2(temp_arr, apply_savgol_filter = True, filename = save_loc_sav)

            # Filtered array
            temp_arr = optical_flow_filters.window_noise_removal(temp_arr, 3)
            temp_arr = optical_flow_filters.expand_intervals(temp_arr, 10, 'max')
            temp_arr = optical_flow_filters.combine_close_actions(temp_arr, 60)
            temp_arr = optical_flow_filters.expand_intervals(temp_arr, 0, 'mean')
            temp_arr[temp_arr < 0.75] = 0
            heat_map_module.numpy_io("write", os.path.join(action_diagram_path, (vid_text + '_filtered.npy')), temp_arr)
            plot_action_diagrams_wp2(temp_arr, apply_savgol_filter = False, filename = save_loc_filtered)

## Checkpoint Complete!##