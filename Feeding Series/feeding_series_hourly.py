# Import statements

import os
from os.path import exists
import glob
import json
import numpy as np
import pandas as pd
import heat_map_module

# Get crop parameters dictionary

if exists("/media/nipek/My Book/Rabbit Research Videos/WP 3.2/Analysis/feeding_series_crop_parameters.json"):
    with open("/media/nipek/My Book/Rabbit Research Videos/WP 3.2/Analysis/feeding_series_crop_parameters.json", "r") as f:
        feeding_series_crop_parameters = json.load(f)
else:
    print('Create feeding series crop parameter dictionary!')

# Calculate feeding behaviour

for dir in glob.glob('/media/nipek/My Book/Rabbit Research Videos/WP 3.2/Analysis/Heatmaps/C*'):
    camera_text = dir.rsplit('/', 1)[1]
    date_time_list = []
    feeding_list = []
    for vid in sorted(glob.glob(os.path.join(dir, '*.npy'))):
        if '020000' not in vid:
            # Date time manipulation
            date_time_text = vid.rsplit('/', 1)[1].rsplit('.', 1)[0].rsplit('.', 1)[1].replace('_', '')
            date_time_object = pd.to_datetime(date_time_text, format='%Y%m%d%H%M%S')
            date_time_list.append(date_time_object)

            # Calculate feeding ratio
            temp_arr = heat_map_module.numpy_io('read', vid)
            station_1_crop_params = feeding_series_crop_parameters[camera_text][0]
            station_2_crop_params = feeding_series_crop_parameters[camera_text][1]
            station_1 = temp_arr[station_1_crop_params[2]:station_1_crop_params[3], station_1_crop_params[0]:station_1_crop_params[1]]
            station_2 = temp_arr[station_2_crop_params[2]:station_2_crop_params[3], station_2_crop_params[0]:station_2_crop_params[1]]
            station_sum = np.sum(station_1) + np.sum(station_2)
            feeding_list.append(station_sum/np.sum(temp_arr))
    save_location = '/media/nipek/My Book/Rabbit Research Videos/WP 3.2/Analysis/Feeding_Time_Series/' + camera_text + '.npy'
    heat_map_module.numpy_io('write', save_location, np.array([date_time_list, feeding_list]))

## Checkpoint Complete!##
