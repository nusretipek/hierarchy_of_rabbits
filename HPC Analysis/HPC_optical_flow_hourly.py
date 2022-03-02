# Import statements
import math
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
from scipy.optimize import curve_fit

# Get crop parameters dictionary

if exists('D:\\Rabbit Research Videos\\WP32_Cycle3\\cage_open_dict.json'):
    with open('D:\\Rabbit Research Videos\\WP32_Cycle3\\cage_open_dict.json', 'r') as f:
        cage_open_dict = json.load(f)
else:
    print('Create cage-open moments dictionary! Else, the feeding series will ignore it')

# Define plot function

def plot_optical_flow_hourly_wp2(arr, time_arr, camera_text, cage_open_dict, apply_savgol_filter = False, filename = None, prediction = None):

    ## Time component
    x_axis = time_arr
    xfmt = md.DateFormatter('%d/%m %H:%M')

    ## Y component and smoothing
    if apply_savgol_filter:
        y_axis = savgol_filter(arr, 11, 3)
    else:
        y_axis = arr

    ## Plot
    fig, ax = plt.subplots(figsize = (42, 18))
    plt.margins(x=0.01)
    fig.subplots_adjust(bottom = 0.2)
    ax.plot(x_axis, y_axis, "-o", color = 'steelblue', markerfacecolor = 'steelblue', markersize = 7, label = 'Action Seconds')
    ax.plot(x_axis, np.zeros(arr.shape[0]) + np.mean(arr), color = 'limegreen', label = 'Mean Action Seconds') #Plot mean line

    ## Add trendline
    z = np.polyfit(np.arange(y_axis.shape[0], dtype = 'float64'), y_axis.astype('float64'), 1)
    p = np.poly1d(z)
    ax.plot(x_axis, p(np.arange(y_axis.shape[0],dtype = 'float64')), "-", color = 'purple', label = 'Trend Line')

    ## Add trendline - Exponential (Custom)
    if prediction is not None:
        ax.plot(x_axis, prediction, "-", color = 'blue', label = 'Exp Trend')

    ## Adjust markers and add legend (cage-open)
    if cage_open_dict is not None:
        cage_open_arr = cage_open_dict[camera_text]
        cage_open_x = []
        cage_open_y = []
        for time_point in cage_open_arr:
            date_time_object = pd.to_datetime(time_point, format='%Y%m%d%H%M%S')
            index = np.where(time_arr == date_time_object)
            cage_open_x.append(time_arr[index])
            cage_open_y.append(y_axis[index])
        ax.plot(cage_open_x, cage_open_y, "s", color = 'crimson', markersize = 10, label = 'Open Cage')
        plt.legend(prop={'size': 8})

    ## Edit plot title and labels
    ax.set(xlabel = 'Time', ylabel = 'Count of Seconds', title = 'Action Count of Seconds Chart' + ' (' + camera_text + ')')
    ax.grid()

    ## X-axis ticks
    ax.xaxis.set_major_formatter(xfmt)
    plt.xticks(rotation = 90)
    x_axis_minor = time_arr
    ax.set_xticks(x_axis_minor)

    ## Save figure
    if filename is not None:
        fig.savefig(filename, bbox_inches='tight')
    else:
        plt.show()
    plt.close()

def exp_sin(x, alpha, a, b, c, d):
    return alpha * (np.power(a,(x))+b) * (np.sin(((2*np.pi/24)*x)+c) + d)

# Hourly aggregate filtered arrays
for dir in sorted(glob.glob('D:\\Rabbit Research Videos\\HPC_Analysis\\WP32_Cycle3\\Action_Diagrams_Intensity_12\\C*')):
    camera_text = dir.rsplit('\\', 1)[1]
    action_analysis_path = os.path.join('D:\\Rabbit Research Videos\\HPC_Analysis\\WP32_Cycle3\\Action_Diagrams_Intensity_12\\', camera_text)
    date_time_list = []
    hourly_action = []
    # Generate time array
    for vid in sorted(glob.glob(os.path.join(dir, '*.npy'))):
        if '020000' not in vid and 'filtered' not in vid:
            # Date time manipulation
            date_time_text = vid.rsplit('\\', 1)[1].rsplit('.', 1)[0].rsplit('.', 1)[1].replace('_', '')
            date_time_object = pd.to_datetime(date_time_text, format='%Y%m%d%H%M%S')
            date_time_list.append(date_time_object)

    # Generate aggregated (hourly) platform usage arrays 1D numpy
    for action_arr in sorted(glob.glob(action_analysis_path + '\\*filtered.npy')):
        if '020000' not in action_arr:
            vid_text = action_arr.rsplit('\\', 1)[1].rsplit('.', 1)[0]
            temp_arr = heat_map_module.numpy_io('read', action_arr)
            hourly_action.append(np.count_nonzero(temp_arr))

    # Curve fit
    x_var = np.arange(0, len(date_time_list), dtype=np.float64)
    y_savgol = savgol_filter(hourly_action, 11, 3)
    y_savgol[0 > y_savgol] = 0
    y_var = np.array(y_savgol, dtype=np.float64)
    popt, pcov = curve_fit(exp_sin, x_var, np.array(hourly_action), maxfev=5000000, method='lm')
    perr = np.sqrt(np.diag(pcov))

    print(camera_text, ':\n \t alpha = %s \t (%s)'
                       ' \n \t a = %s \t (%s)'
                       ' \n \t b = %s \t (%s)'
                       ' \n \t c = %s \t (%s)'
                       ' \n \t d = %s \t (%s) \n'
          % (popt[0], perr[0],
             popt[1], perr[1],
             popt[2], perr[2],
             popt[3], perr[3],
             popt[4], perr[4]))

    predictions = []
    for x in x_var:
        predictions.append(exp_sin(x,popt[0],popt[1],popt[2], popt[3], popt[4]))

    df = pd.DataFrame({'Timestamp': date_time_list, 'Original_Value': hourly_action, 'Savgol_Filter_Value': y_savgol})
    df.to_csv('D:\\Rabbit Research Videos\\HPC_Analysis\\WP32_Cycle3\\Action_Diagrams_Intensity_12\\Hourly_Sequences\\' + camera_text + '.csv')

    # Plot and save
    save_location_reg = 'D:\\Rabbit Research Videos\\HPC_Analysis\\WP32_Cycle3\\Action_Diagrams_Intensity_12\\Hourly_Sequences\\' + camera_text + '_plot.jpg'
    save_location_savgol = 'D:\\Rabbit Research Videos\\HPC_Analysis\\WP32_Cycle3\\Action_Diagrams_Intensity_12\\Hourly_Sequences\\' + camera_text + '_savgol_plot.jpg'
    plot_optical_flow_hourly_wp2(np.array(hourly_action), np.array(date_time_list),  camera_text,
                                  None, apply_savgol_filter = False, filename = save_location_reg, prediction = None) ## ! Cage open deleted!!!!!!
    plot_optical_flow_hourly_wp2(np.array(hourly_action), np.array(date_time_list),  camera_text,
                                  None , apply_savgol_filter = True, filename = save_location_savgol, prediction = np.array(predictions)) ## ! Cage open deleted!!!!!!

## Checkpoint Complete!##