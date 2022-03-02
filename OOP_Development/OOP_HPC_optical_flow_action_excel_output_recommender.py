# Import statements

import os
import glob
import numpy as np
import pandas as pd
import heat_map_module
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Video sampling times
sample_bool = False
times_arr = ['20210810_13',
             '20210810_14',
             '20210810_15',
             '20210810_16',
             '20210810_17',
             '20210810_18',
             '20210810_19',
             '20210810_20',
             '20210810_21',
             '20210810_22',
             '20210810_23',
             '20210811_00',
             '20210811_01',
             '20210811_02',
             '20210811_03',
             '20210811_04',
             '20210811_05',
             '20210811_06',
             '20210811_07',
             '20210811_08',
             '20210811_09',
             '20210811_10',
             '20210811_11',
             '20210811_12',
             '20210812_13',
             '20210812_14',
             '20210812_15',
             '20210812_16',
             '20210812_17',
             '20210812_18',
             '20210812_19',
             '20210812_20',
             '20210812_21',
             '20210812_22',
             '20210812_23',
             '20210813_00',
             '20210813_01',
             '20210813_02',
             '20210813_03',
             '20210813_04',
             '20210813_05',
             '20210813_06',
             '20210813_07',
             '20210813_08',
             '20210813_09',
             '20210813_10',
             '20210813_11',
             '20210813_12',
             '20210816_09',
             '20210816_10',
             '20210816_11',
             '20210816_12',
             '20210816_13',
             '20210816_14',
             '20210816_15',
             '20210816_16',
             '20210816_17',
             '20210816_18',
             '20210816_19',
             '20210816_20',
             '20210816_21',
             '20210816_22',
             '20210816_23',
             '20210817_00',
             '20210817_01',
             '20210817_02',
             '20210817_03',
             '20210817_04',
             '20210817_05',
             '20210817_06',
             '20210817_07',
             '20210817_08',
             '20210819_08',
             '20210819_09',
             '20210819_10',
             '20210819_11',
             '20210819_12',
             '20210819_13',
             '20210819_14',
             '20210819_15',
             '20210819_16',
             '20210819_17',
             '20210819_18',
             '20210819_19',
             '20210819_20',
             '20210819_21',
             '20210819_22',
             '20210819_23',
             '20210820_00',
             '20210820_01',
             '20210820_02',
             '20210820_03',
             '20210820_04',
             '20210820_05',
             '20210820_06',
             '20210820_07']

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

for dir in sorted(glob.glob('D:\\Rabbit Research Videos\\HPC_Analysis\\WP32_Cycle3\\Action_Diagrams\\C*')):
    camera_text = dir.rsplit('\\', 1)[1]
    action_diagram_path = os.path.join('D:\\Rabbit Research Videos\\HPC_Analysis\\WP32_Cycle3\\Action_Diagrams\\', camera_text)

    # Create action diagram directories
    if not os.path.exists(action_diagram_path):
        os.mkdir(action_diagram_path)

    # Generate action diagrams from 1D numpy arrays
    master_df = pd.DataFrame(columns=["Video_Name", "Action_Start", "Action_End", "Duration", "Intensity"])
    for action_arr in sorted(glob.glob(action_diagram_path + '\\*filtered.npy')):
        if '020000' not in action_arr:
                vid_text = action_arr.rsplit('\\', 1)[1].rsplit('.', 1)[0]
                if sample_bool:
                    if (vid_text[6:17] in times_arr):
                        temp_arr = heat_map_module.numpy_io('read', action_arr)
                        master_df = master_df.append(get_action_moments_wp2(temp_arr, vid_text.rsplit("_", 1)[0]))
                else:
                    temp_arr = heat_map_module.numpy_io('read', action_arr)
                    master_df = master_df.append(get_action_moments_wp2(temp_arr, vid_text.rsplit("_", 1)[0]))
    master_df.to_excel('D:\\Rabbit Research Videos\\HPC_Analysis\\WP32_Cycle3\\Action_Diagrams\\Action_Moments_Excel_Recommender\\' + camera_text + '.xlsx',
                       index=False)

## Checkpoint Complete!##