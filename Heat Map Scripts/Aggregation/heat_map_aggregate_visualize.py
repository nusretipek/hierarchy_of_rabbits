# Import statements

import heat_map_module
import cv2
import glob
import numpy as np

# Visualize aggregate heatmaps

for dir in glob.glob('/media/nipek/My Book/Rabbit Research Videos/WP 3.2/Analysis/Master_Heatmaps/*'):
    for vid in glob.glob(dir + '/*.npy'):
        location_name = vid.rsplit('/', 1)[0]
        vid_name = vid.rsplit('/', 1)[1].rsplit('.', 1)[0]
        temp_arr = heat_map_module.numpy_io('read', vid)

        # Binary
        cv2.imwrite(location_name + '/' + vid_name + '.jpg', temp_arr * 255)

        # RGBA - Red
        rgba = cv2.cvtColor(np.full((temp_arr.shape[0], temp_arr.shape[1], 3), (0, 0, 255), np.dtype('uint8')), cv2.COLOR_RGB2RGBA)
        rgba[:, :, 3] = temp_arr * 255
        cv2.imwrite(location_name + '/' + vid_name + '_rgba.png', rgba)

## Checkpoint Complete!##
