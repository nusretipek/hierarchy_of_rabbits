# Import statements
import os
import glob
import numpy as np
import cv2
import random

# Global parameters
save_location = '/media/nipek/My Book/Rabbit Research Videos/WP 3.2/Analysis/Detectron 2/Random_Images_Last_Day'
total_images = 21
videos_list = []
counter = 1


# Function to get random image from a video
def get_random_frame(vid):
    cap = cv2.VideoCapture(vid)
    cap.set(1, random.randint(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))
    success, image = cap.read()
    return image


# Read all videos in a Numpy Array
for dir in sorted(glob.glob('/media/nipek/My Book/Rabbit Research Videos/WP 3.2/C*')):
    for vid in sorted(glob.glob(dir + '/*')):
        if '020000' not in vid:
            if vid.rsplit("_", 1)[0][-2:] in ['08']:  # ['29', '30', '01', '02']
                videos_list.append(vid)
videos_list = np.array(videos_list)

# Get random videos with replacement
np.random.seed(0)
random_videos_list = np.random.choice(videos_list, size=total_images, replace=True)
for vid in random_videos_list:
    random_image = get_random_frame(vid)
    cv2.imwrite(os.path.join(save_location, (str(counter) + '.jpg')), random_image)
    counter += 1

## Checkpoint Complete!##
