# Import statements

import heat_map_module
from os.path import exists
from os import walk
import cv2
import glob
import json

# Generate and save std and peakness

for np_file in glob.glob('/media/ricky/My Book/Rabbit Research Videos/WP 3.2/Analysis/*[arr].npy'):
    camera_text = np_file.rsplit('/', 1)[1]
    camera_text_arr = camera_text.rsplit('.', 1)[0]
    temp_arr = heat_map_module.numpy_io('read', np_file)
    temp_pixel_std = heat_map_module.pixel_std(temp_arr)
    temp_peak_sharpness = heat_map_module.get_peak_sharpness(temp_arr)
    heat_map_module.numpy_io('write', '/media/ricky/My Book/Rabbit Research Videos/WP 3.2/Analysis/' + camera_text_arr + '_std.npy', temp_pixel_std)
    heat_map_module.numpy_io('write', '/media/ricky/My Book/Rabbit Research Videos/WP 3.2/Analysis/' + camera_text_arr + '_peakness.npy', temp_peak_sharpness)

## Checkpoint Complete! ##