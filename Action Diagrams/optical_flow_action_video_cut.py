# Import statements
import os
from os.path import exists
import glob
import numpy as np
import pandas as pd
import json
import cv2
import heat_map_module
import warnings
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
warnings.simplefilter(action='ignore', category=FutureWarning)

# Define Output Function

def get_action_moments_wp2(arr, vid_name):
    temp_df = pd.DataFrame(columns=["Video_Name", "Action_Start", "Action_End", "Duration", "Intensity"])
    action_indices = np.where(np.diff(arr) != 0)[0]
    if arr[0] > 0:
        action_indices = np.insert(action_indices, 0, 0)
    if arr[len(arr)-1] > 0:
        action_indices = np.insert(action_indices, len(action_indices), len(arr)-2)
    for i in range(int(len(action_indices)/2)):
        temp_df = temp_df.append({"Video_Name": vid_name,
                                  "Action_Start": str(int(action_indices[2*i]/60)).zfill(2) + ":" + str(int(action_indices[2*i]%60)).zfill(2),
                                  "Action_End": str(int(action_indices[(2*i)+1]/60)).zfill(2) + ":" + str(int(action_indices[(2*i)+1]%60)).zfill(2),
                                  "Duration": int(action_indices[(2*i)+1]) - int(action_indices[2*i]),
                                  "Intensity": round(arr[action_indices[2*i]+1], 1)}, ignore_index=True)
    return temp_df

# Create a loop function to save action clips

for dir in sorted(glob.glob('D:\\Rabbit Research Videos\\WP 3.2\\C*')):
    camera_text = dir.rsplit('\\', 1)[1]
    action_diagram_path = os.path.join('D:\\Rabbit Research Videos\\HPC_Analysis\\WP32\\Action_Diagrams\\', camera_text)
    save_location = os.path.join('D:\\Rabbit Research Videos\\HPC_Analysis\\WP32\\Action_Video_Clips', camera_text)

    # Create action diagram directories
    if not os.path.exists(save_location):
        os.mkdir(save_location)

    # Generate action diagrams from 1D numpy arrays
    master_df = pd.DataFrame(columns=["Video_Name", "Action_Start", "Action_End", "Duration", "Intensity"])
    for action_arr in sorted(glob.glob(action_diagram_path + '\\*filtered.npy')):
        if '020000' not in action_arr:
                vid_text = action_arr.rsplit('\\', 1)[1].rsplit('.', 1)[0]
                temp_arr = heat_map_module.numpy_io('read', action_arr)
                master_df = master_df.append(get_action_moments_wp2(temp_arr, vid_text.rsplit("_", 1)[0]))

    master_df['Action_Start'] = pd.to_datetime(master_df['Action_Start'], format='%M:%S')
    master_df['Action_End'] = pd.to_datetime(master_df['Action_End'], format='%M:%S')

    counter = 1
    for index, row in master_df.iterrows():
        ffmpeg_extract_subclip(os.path.join(dir, row['Video_Name'] + '.mp4'),
                                60*(row['Action_Start'].minute)+row['Action_Start'].second,
                                60*(row['Action_End'].minute)+row['Action_End'].second,
                                targetname=os.path.join(save_location, 'Action_' + str(counter) + '.mp4'))
        counter += 1

## Checkpoint Complete!##