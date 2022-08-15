# Import statements

import heat_map_module
from os.path import exists
import glob
import cv2
import json

# Create a crop parameter dictionary

if exists("/media/ricky/My Book/Rabbit Research Videos/WP 3.2/Analysis/crop_parameters.json"):
    with open("/media/ricky/My Book/Rabbit Research Videos/WP 3.2/Analysis/crop_parameters.json", "r") as f:
        crop_parameters = json.load(f)
else:
    with open("/media/ricky/My Book/Rabbit Research Videos/WP 3.2/Analysis/crop_parameters.json", "w") as f:
        json.dump({"Ex_cam": [1, 2, 3, 4], "Ex_cam2": [1, 2, 3, 4]}, f, indent=4)

# Get crop parameter inspection images (Void)

for dir in glob.glob('/media/ricky/My Book/Rabbit Research Videos/WP 3.2/C*'):
    camera_text = dir.rsplit('/', 1)[1]
    camera_number = camera_text.rsplit(' ', 1)[1]
    if int(camera_number) < 10:
        vid_file = '/media/ricky/My Book/Rabbit Research Videos/WP 3.2/' + camera_text + '/kon0' + camera_number + '.20210629_170000.mp4'
    else:
        vid_file = '/media/ricky/My Book/Rabbit Research Videos/WP 3.2/' + camera_text + '/kon' + camera_number + '.20210629_170000.mp4'
    image = heat_map_module.get_initial_frame(vid_file)
    cv2.imwrite(
        '/media/ricky/My Book/Rabbit Research Videos/WP 3.2/Analysis/Camera_' + camera_number + '_initial' + '.jpg',
        image)

# Get cropped images (Check)

for dir in glob.glob('/media/ricky/My Book/Rabbit Research Videos/WP 3.2/C*'):
    camera_text = dir.rsplit('/', 1)[1]
    camera_number = camera_text.rsplit(' ', 1)[1]
    if int(camera_number) < 10:
        vid_file = '/media/ricky/My Book/Rabbit Research Videos/WP 3.2/' + camera_text + '/kon0' + camera_number + '.20210629_170000.mp4'
    else:
        vid_file = '/media/ricky/My Book/Rabbit Research Videos/WP 3.2/' + camera_text + '/kon' + camera_number + '.20210629_170000.mp4'
    image = heat_map_module.get_initial_frame(vid_file, crop_check=True, crop_params=crop_parameters[camera_text])
    cv2.imwrite(
        '/media/ricky/My Book/Rabbit Research Videos/WP 3.2/Analysis/Camera_' + camera_number + '_cropped' + '.jpg',
        image)

## Checkpoint Complete! ##
