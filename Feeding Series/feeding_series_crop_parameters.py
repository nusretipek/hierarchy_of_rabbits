# Import statements

import heat_map_module
from os.path import exists
import glob
import cv2
import json

# Create a crop parameter dictionary

if exists("/media/nipek/My Book/Rabbit Research Videos/WP 3.2/Analysis/feeding_series_crop_parameters.json"):
    with open("/media/nipek/My Book/Rabbit Research Videos/WP 3.2/Analysis/feeding_series_crop_parameters.json", "r") as f:
        feeding_series_crop_parameters = json.load(f)
else:
    with open("/media/nipek/My Book/Rabbit Research Videos/WP 3.2/Analysis/feeding_series_crop_parameters.json",
              "w") as f:
        json.dump({"Camera 1": [1, 2, 3, 4], "Camera 2": [1, 2, 3, 4]}, f, indent=4)

# Get crop parameter inspection images (Void)

for dir in glob.glob('/media/nipek/My Book/Rabbit Research Videos/WP 3.2/Analysis/Heatmaps/C*'):
    camera_text = dir.rsplit('/', 1)[1]
    camera_number = camera_text.rsplit(' ', 1)[1]
    sample_arr = heat_map_module.numpy_io("read", glob.glob(dir + "/*20210629_170000.npy")[0])
    station_1_crop_params = feeding_series_crop_parameters[camera_text][0]
    station_2_crop_params = feeding_series_crop_parameters[camera_text][1]
    station_1_image = sample_arr[station_1_crop_params[2]:station_1_crop_params[3],station_1_crop_params[0]:station_1_crop_params[1]]
    station_2_image = sample_arr[station_2_crop_params[2]:station_2_crop_params[3],station_2_crop_params[0]:station_2_crop_params[1]]
    cv2.imwrite('/media/nipek/My Book/Rabbit Research Videos/WP 3.2/Analysis/Feeding_Station_Crop_Check/Camera_' + camera_number + '_feeding_station_1' + '.jpg',
        station_1_image * 255)
    cv2.imwrite('/media/nipek/My Book/Rabbit Research Videos/WP 3.2/Analysis/Feeding_Station_Crop_Check/Camera_' + camera_number + '_feeding_station_2' + '.jpg',
        station_2_image * 255)

## Checkpoint Complete! ##