# Import statements

import os
from os.path import exists
import glob
import numpy as np
import pandas as pd
import json
import heat_map_module

# Get crop parameters dictionary

if exists("/media/nipek/My Book/Rabbit Research Videos/WP 3.2/Analysis/cage_open_dict.json"):
    with open("/media/nipek/My Book/Rabbit Research Videos/WP 3.2/Analysis/cage_open_dict.json", "r") as f:
        cage_open_dict = json.load(f)
else:
    print('Create cage-open moments dictionary! Else, the feeding series will ignore it')

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

for dir in sorted(glob.glob('/media/nipek/My Book/Rabbit Research Videos/WP 3.2/C*')):
    camera_text = dir.rsplit('/', 1)[1]
    action_diagram_path = os.path.join('/media/nipek/My Book/Rabbit Research Videos/WP 3.2/Analysis/Action_Diagrams', camera_text)

    # Create action diagram directories
    if not os.path.exists(action_diagram_path):
        os.mkdir(action_diagram_path)

    # Generate action diagrams from 1D numpy arrays
    master_df = pd.DataFrame(columns=["Video_Name", "Action_Start", "Action_End", "Duration", "Intensity"])
    for action_arr in sorted(glob.glob(action_diagram_path + '/*filtered.npy')):
        if '020000' not in action_arr:
                vid_text = action_arr.rsplit('/', 1)[1].rsplit('.', 1)[0]
                temp_arr = heat_map_module.numpy_io('read', action_arr)
                master_df = master_df.append(get_action_moments_wp2(temp_arr, vid_text.rsplit("_", 1)[0]))
    master_df.to_excel('/media/nipek/My Book/Rabbit Research Videos/WP 3.2/Analysis/Action_Diagrams/Action_Moments_Excel_Recommender/' + camera_text + '.xlsx',
                       index=False)

## Checkpoint Complete!##