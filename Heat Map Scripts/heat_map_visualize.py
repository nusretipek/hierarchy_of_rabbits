# Import statements

import os
import heat_map_module
import cv2
import glob

# Visualize heatmaps

for dir in glob.glob('/media/nipek/My Book/Rabbit Research Videos/WP 3.2/C*'):
    camera_text = dir.rsplit('/', 1)[1]
    heatmap_path = os.path.join('/media/nipek/My Book/Rabbit Research Videos/WP 3.2/Analysis/Heatmaps', camera_text)
    for vid in glob.glob(dir + '/*'):
        vid_name = vid.rsplit('/', 1)[1].rsplit('.',1)[0]
        cap = cv2.VideoCapture(vid)
        if int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) > 80000:
            temp_arr = heat_map_module.numpy_io('read', os.path.join(heatmap_path, (vid_name + '.npy')))
            cv2.imwrite(os.path.join(heatmap_path, (vid_name + '_heatmap.jpg')), temp_arr*255)

## Checkpoint Complete!##