# Import statements

import os
import heat_map_module
import cv2
import glob
import numpy as np
from skimage.transform import resize
from skimage.transform import rotate
from itertools import combinations

# Define experiment clusters
camera_groups = {
    "Control": ['Camera 10', 'Camera 22', 'Camera 25'],
    "Alfalfa": ['Camera 5', 'Camera 12', 'Camera 24'],
    "Wood_Blocks": ['Camera 8', 'Camera 11', 'Camera 23'],
    "Alfalfa_Wood_Blocks": ['Camera 1', 'Camera 9', 'Camera 21']
}

# Get resize parameters

resize_param_low_w = 9999
resize_param_high_w = 0
resize_param_low_h = 9999
resize_param_high_h = 0

for key in camera_groups:
    for camera in camera_groups[key]:
        heat_map_location = os.path.join('/media/nipek/My Book/Rabbit Research Videos/WP 3.2/Analysis/Master_Heatmaps/72_Hours', camera)
        temp_arr = heat_map_module.numpy_io('read', heat_map_location + '.npy')
        if (temp_arr.shape[0] < resize_param_low_h):
            resize_param_low_h  = temp_arr.shape[0]
        if (temp_arr.shape[0] > resize_param_high_h):
            resize_param_high_h = temp_arr.shape[0]
        if (temp_arr.shape[1] < resize_param_low_w):
            resize_param_low_w = temp_arr.shape[1]
        if (temp_arr.shape[1] > resize_param_high_w):
            resize_param_high_w = temp_arr.shape[1]

print(resize_param_low_h, resize_param_high_h)
print(resize_param_low_w, resize_param_high_w)

# Visualize heatmaps

## Low Resize

for key in camera_groups:
    counter = 0
    aggregate_arr = None
    for camera in camera_groups[key]:

        # Read master heat maps
        heat_map_location = os.path.join('/media/nipek/My Book/Rabbit Research Videos/WP 3.2/Analysis/Master_Heatmaps/72_Hours', camera)
        temp_arr = heat_map_module.numpy_io('read', heat_map_location + '.npy')

        # Rotate if necessary
        if camera in ['Camera 9', 'Camera 21']:
            temp_arr = rotate(temp_arr, 180)

        # Resize the heat maps (All cages are different)
        temp_arr_resized = resize(temp_arr, (resize_param_low_h, resize_param_low_w))

        # Add heatmaps
        if (aggregate_arr is None):
            aggregate_arr = temp_arr_resized
        else:
            aggregate_arr += temp_arr_resized

        # Increment the counter
        counter += 1

    # Divide and save

    aggregate_arr /= counter
    save_folder_location = '/media/nipek/My Book/Rabbit Research Videos/WP 3.2/Analysis/Master_Heatmaps/72_Hours_Difference/Low_Resize'
    heat_map_module.numpy_io('write', os.path.join(save_folder_location, (key + '_mean.npy')), aggregate_arr)
    cv2.imwrite(os.path.join(save_folder_location, (key + '_mean.png')), aggregate_arr*255)

## High Resize

for key in camera_groups:
    counter = 0
    aggregate_arr = None
    for camera in camera_groups[key]:

        # Read master heat maps
        heat_map_location = os.path.join('/media/nipek/My Book/Rabbit Research Videos/WP 3.2/Analysis/Master_Heatmaps/72_Hours', camera)
        temp_arr = heat_map_module.numpy_io('read', heat_map_location + '.npy')

        # Rotate if necessary
        if camera in ['Camera 9', 'Camera 21']:
            temp_arr = rotate(temp_arr, 180)

        # Resize the heat maps (All cages are different)
        temp_arr_resized = resize(temp_arr, (resize_param_high_h, resize_param_high_w))

        # Add heatmaps
        if (aggregate_arr is None):
            aggregate_arr = temp_arr_resized
        else:
            aggregate_arr += temp_arr_resized

        # Increment the counter
        counter += 1

    # Divide and save

    aggregate_arr /= counter
    save_folder_location = '/media/nipek/My Book/Rabbit Research Videos/WP 3.2/Analysis/Master_Heatmaps/72_Hours_Difference/High_Resize'
    heat_map_module.numpy_io('write', os.path.join(save_folder_location, (key + '_mean.npy')), aggregate_arr)
    cv2.imwrite(os.path.join(save_folder_location, (key + '_mean.png')), aggregate_arr*255)

# Difference heatmaps
save_folder_location_low = '/media/nipek/My Book/Rabbit Research Videos/WP 3.2/Analysis/Master_Heatmaps/72_Hours_Difference/Low_Resize'
save_folder_location_high = '/media/nipek/My Book/Rabbit Research Videos/WP 3.2/Analysis/Master_Heatmaps/72_Hours_Difference/High_Resize'
low_heat_maps = []
high_heat_maps = []

for low_arr in glob.glob(save_folder_location_low + '/*.npy'):
    low_heat_maps.append(low_arr)

for high_arr in glob.glob(save_folder_location_high + '/*.npy'):
    high_heat_maps.append(high_arr)

low_combiniations = list(combinations(low_heat_maps, 2))
high_combiniations = list(combinations(high_heat_maps, 2))

save_folder_location_low_diff = '/media/nipek/My Book/Rabbit Research Videos/WP 3.2/Analysis/Master_Heatmaps/72_Hours_Difference/Low_Resize/Differences'
save_folder_location_high_diff = '/media/nipek/My Book/Rabbit Research Videos/WP 3.2/Analysis/Master_Heatmaps/72_Hours_Difference/High_Resize/Differences'

for pair in low_combiniations:
    temp_arr_1 = heat_map_module.numpy_io('read', pair[0])
    temp_arr_2 = heat_map_module.numpy_io('read', pair[1])

    # Difference operation
    diff_arr = np.abs(temp_arr_1 - temp_arr_2)
    diff_arr[diff_arr <= 0.25]  = 0
    diff_arr[diff_arr >= 0.25]  = 1

    unique_save_name = pair[0].rsplit('/', 1)[1].rsplit('.', 1)[0].rsplit('_', 1)[0] + "_and_"  + pair[1].rsplit('/', 1)[1].rsplit('.', 1)[0].rsplit('_', 1)[0]
    heat_map_module.numpy_io('write', os.path.join(save_folder_location_low_diff, (unique_save_name + '.npy')), diff_arr)
    cv2.imwrite(os.path.join(save_folder_location_low_diff, (unique_save_name + '.png')), diff_arr*255)

for pair in high_combiniations:
    temp_arr_1 = heat_map_module.numpy_io('read', pair[0])
    temp_arr_2 = heat_map_module.numpy_io('read', pair[1])

    # Difference operation
    diff_arr = np.abs(temp_arr_1 - temp_arr_2)
    diff_arr[diff_arr <= 0.25]  = 0
    diff_arr[diff_arr >= 0.25]  = 1

    unique_save_name = pair[0].rsplit('/', 1)[1].rsplit('.', 1)[0].rsplit('_', 1)[0] + "_and_"  + pair[1].rsplit('/', 1)[1].rsplit('.', 1)[0].rsplit('_', 1)[0]
    heat_map_module.numpy_io('write', os.path.join(save_folder_location_high_diff, (unique_save_name + '.npy')), diff_arr)
    cv2.imwrite(os.path.join(save_folder_location_high_diff, (unique_save_name + '.png')), diff_arr*255)

## Checkpoint Complete!##