# Import statements

import os
import cv2
import glob
import pandas as pd
pd.set_option('display.width', 320)
pd.set_option('display.max_columns', 5)

# Get dataframes for heat map aggregation

for dir in glob.glob('/media/nipek/My Book/Rabbit Research Videos/WP 3.2/C*'):
    camera_text = dir.rsplit('/', 1)[1]
    heatmap_path = os.path.join('/media/nipek/My Book/Rabbit Research Videos/WP 3.2/Analysis/Heatmaps', camera_text)
    master_list_pandas = []
    for heat_map in sorted(glob.glob(heatmap_path + '/*.npy')):

        # Date time manipulation
        date_time_text = heat_map.rsplit('/', 1)[1].rsplit('.', 1)[0].rsplit('.', 1)[1].replace('_', '')
        date_time_object = pd.to_datetime(date_time_text, format='%Y%m%d%H%M%S')

        # Parse vid location & Get frame count
        vid_location = ("/media/nipek/My Book/Rabbit Research Videos/WP 3.2/" +
                        (heat_map.rsplit('/', 2)[1] + '/' + heat_map.rsplit('/', 1)[1]).rsplit('.', 1)[0] + '.mp4')
        cap = cv2.VideoCapture(vid_location)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Create list for pandas
        temp_list = [heat_map, date_time_object, vid_location, frame_count]
        master_list_pandas.append(temp_list)

    df = pd.DataFrame(master_list_pandas, columns=['heat_map_location', 'date_time', 'vid_location', 'frame_count'])
    df.to_csv(('/media/nipek/My Book/Rabbit Research Videos/WP 3.2/Analysis/Master_Heatmaps/Dataframes/' + camera_text + '.csv'),
              index=False)

## Checkpoint Complete!##
