# Import statements

import heat_map_module
from os.path import exists
from os import walk
import cv2
import glob
import json

# Generate and save masks

for np_file in glob.glob('/media/ricky/My Book/Rabbit Research Videos/WP 3.2/Analysis/*std.npy'):
    camera_text = np_file.rsplit('/', 1)[1]
    camera_text_arr = camera_text.rsplit('.', 1)[0]
    camera_text_name = camera_text.rsplit('_', 2)[0]
    if (camera_text_name in ["Camera_1", "Camera_1_night"]):
        temp_arr = heat_map_module.numpy_io('read', '/media/ricky/My Book/Rabbit Research Videos/WP 3.2/Analysis/' + camera_text_name + '_arr.npy')
        temp_std = heat_map_module.numpy_io('read', '/media/ricky/My Book/Rabbit Research Videos/WP 3.2/Analysis/' + camera_text_name + '_arr_std.npy')
        temp_peakness = heat_map_module.numpy_io('read', '/media/ricky/My Book/Rabbit Research Videos/WP 3.2/Analysis/' + camera_text_name + '_arr_peakness.npy')
        temp_mask = heat_map_module.get_mask_wp3_2(temp_arr, temp_std, temp_peakness)
        heat_map_module.numpy_io('write', '/media/ricky/My Book/Rabbit Research Videos/WP 3.2/Analysis/' + camera_text_name + '_mask.npy', temp_mask)
        cv2.imwrite('/media/ricky/My Book/Rabbit Research Videos/WP 3.2/Analysis/' + camera_text_name + '_mask.jpg', temp_mask)

# Number 10: STD 25 and 20 adjusted
# Number 22 and 23: STD 30 and 25 adjusted + peakness 0.003
# Number 1:
#  mask[np.logical_and(np.logical_and((arr > 50), (frame_std <= 35)), (peak > 0.02))] = 0
#  mask[np.logical_and(np.logical_or((peak < 0.045), (frame_std > 23)), (arr <= 200))] = 255
# All others:
##  mask[np.logical_and(np.logical_and((arr > 50), (frame_std <= 35)), (peak > 0.001))] = 0
##  mask[np.logical_and(np.logical_or((peak < 0.005), (frame_std > 30)), (arr <= 200))] = 255

## Checkpoint Complete! ##