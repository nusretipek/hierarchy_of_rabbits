# Import statements

import os
import sys
import heat_map_module
from os.path import exists
from os import walk
import cv2
import glob
import json
import multiprocessing as mp

# Get crop parameter dictionary

if exists("/media/nipek/My Book/Rabbit Research Videos/WP 3.2/Analysis/crop_parameters.json"):
    with open("/media/nipek/My Book/Rabbit Research Videos/WP 3.2/Analysis/crop_parameters.json", "r") as f:
        crop_parameters = json.load(f)
else:
    print('Create crop parameter dictionary!')

# Generate and save hourly heatmaps

for dir in glob.glob('/media/nipek/My Book/Rabbit Research Videos/WP 3.2/C*'):
    camera_text = dir.rsplit('/', 1)[1]
    if camera_text in ['Camera 1', 'Camera 10']:
        continue
    heatmap_path = os.path.join('/media/nipek/My Book/Rabbit Research Videos/WP 3.2/Analysis/Heatmaps', camera_text)
    mask_path = os.path.join('/media/nipek/My Book/Rabbit Research Videos/WP 3.2/Analysis/Final_Masks', (camera_text.replace(' ', '_') + '_mask.jpg'))

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
    argument_arr = []
    for vid in glob.glob(dir + '/*'):
        vid_name = vid.rsplit('/', 1)[1].rsplit('.',1)[0]
        argument_arr.append((vid, mask_np, os.path.join(heatmap_path, (vid_name + '.npy')), crop_parameters[camera_text], vid_name))

    with mp.Pool(6) as p:
        p.starmap(heat_map_module.get_heat_map_wp2_pool, argument_arr)

## Checkpoint Complete!##