# Import statements

import os
import heat_map_module
from os.path import exists
import cv2
import glob
import json

# Get crop parameter dictionary

if exists("/media/ricky/My Book/Rabbit Research Videos/WP 3.2/Analysis/crop_parameters.json"):
    with open("/media/ricky/My Book/Rabbit Research Videos/WP 3.2/Analysis/crop_parameters.json", "r") as f:
        crop_parameters = json.load(f)
else:
    print('Create crop parameter dictionary!')

# Generate and save hourly heatmaps

for dir in glob.glob('/media/ricky/My Book/Rabbit Research Videos/WP 3.2/Camera 1'):
    camera_text = dir.rsplit('/', 1)[1]
    heatmap_path = os.path.join('/media/ricky/My Book/Rabbit Research Videos/WP 3.2/Analysis/Heatmaps', camera_text)
    mask_path = os.path.join('/media/ricky/My Book/Rabbit Research Videos/WP 3.2/Analysis/Final_Masks', (camera_text.replace(' ', '_') + '_mask.jpg'))

    # Create heatmap directories
    if not os.path.exists(heatmap_path):
        os.mkdir(heatmap_path)

    # Check and read masks
    if not os.path.exists(mask_path):
        print('No mask detected:', camera_text)
    else:
        print('Mask detected:', camera_text)
        mask_np = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # Generate heatmap arrays 2D
    for vid in glob.glob(dir + '/*'):
        vid_name = vid.rsplit('/', 1)[1].rsplit('.',1)[0]
        temp_arr = heat_map_module.get_heat_map_wp2(vid, mask_np, crop_parameters[camera_text], vid_name)
        heat_map_module.numpy_io('write', os.path.join(heatmap_path, (vid_name + '.npy')), temp_arr)

## Checkpoint Complete!##