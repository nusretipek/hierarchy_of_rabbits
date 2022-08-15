# Import statements

from os.path import exists
import glob
import numpy as np
import pandas as pd
import json
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

def plot_feeding_series_wp2(arr, camera_text, cage_open_dict, apply_savgol_filter=False, filename=None):
    ## Time component
    x_axis = arr[0]
    xfmt = md.DateFormatter('%d/%m %H:%M')

    ## Y component and smoothing
    if apply_savgol_filter:
        y_axis = savgol_filter(arr[1], 11, 3)
    else:
        y_axis = arr[1]

    ## Plot
    fig, ax = plt.subplots(figsize=(20, 8))
    fig.subplots_adjust(bottom=0.2)
    ax.plot(x_axis, y_axis, "-o", color='steelblue', markerfacecolor='steelblue', markersize=7, label='Feeding Ratio')
    ax.plot(x_axis, np.zeros(arr[1].shape[0]) + np.mean(arr[1]), color='limegreen',
            label='Mean Feeding Ratio')  # Plot mean line

    ## Add trendline
    z = np.polyfit(np.arange(y_axis.shape[0], dtype='float64'), y_axis.astype('float64'), 1)
    p = np.poly1d(z)
    ax.plot(x_axis, p(np.arange(y_axis.shape[0], dtype='float64')), "-", color='purple', label='Trend Line')

    ## Adjust markers and add legend (cage-open)
    cage_open_arr = cage_open_dict[camera_text]
    cage_open_x = []
    cage_open_y = []
    for time_point in cage_open_arr:
        date_time_object = pd.to_datetime(time_point, format='%Y%m%d%H%M%S')
        index = np.where(arr[0] == date_time_object)
        cage_open_x.append(arr[0][index])
        cage_open_y.append(y_axis[index])
    ax.plot(cage_open_x, cage_open_y, "s", color='crimson', markersize=10, label='Open Cage')
    plt.legend(prop={'size': 8})

    ## Edit plot title and labels
    ax.set(xlabel='Time', ylabel='Feeding Ratio', title='Feeding Station Occupancy Chart' + ' (' + camera_text + ')')
    ax.grid()

    ## X-axis ticks
    ax.xaxis.set_major_formatter(xfmt)
    plt.xticks(rotation=90)
    x_axis_minor = arr[0]
    ax.set_xticks(x_axis_minor)

    ## Save figure
    if filename is not None:
        fig.savefig(filename)
    else:
        plt.show()


# Plot Upper Platform Usage

for feeding_series in glob.glob(
        '/media/nipek/My Book/Rabbit Research Videos/WP 3.2/Analysis/Feeding_Time_Series/*.npy'):
    camera_text = feeding_series.rsplit('/', 1)[1].rsplit('.', 1)[0]

    # Load numpy array
    with open(feeding_series, 'rb') as f:
        temp_arr = np.load(f, allow_pickle=True)

    # Plot and save
    save_location_reg = '/media/nipek/My Book/Rabbit Research Videos/WP 3.2/Analysis/Feeding_Time_Series/' + camera_text + '_plot.jpg'
    save_location_savgol = '/media/nipek/My Book/Rabbit Research Videos/WP 3.2/Analysis/Feeding_Time_Series/' + camera_text + '_savgol_plot.jpg'
    plot_feeding_series_wp2(temp_arr, camera_text, cage_open_dict,
                            apply_savgol_filter=False, filename=save_location_reg)
    plot_feeding_series_wp2(temp_arr, camera_text, cage_open_dict,
                            apply_savgol_filter=True, filename=save_location_savgol)

## Checkpoint Complete!##
