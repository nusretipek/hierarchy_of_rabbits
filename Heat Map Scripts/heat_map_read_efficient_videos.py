import heat_map_module
from os.path import exists
from os import walk
import glob
import json

# Get crop parameter dictionary

if exists("/media/ricky/My Book/Rabbit Research Videos/WP 3.2/Analysis/crop_parameters.json"):
    with open("/media/ricky/My Book/Rabbit Research Videos/WP 3.2/Analysis/crop_parameters.json", "r") as f:
        crop_parameters = json.load(f)
else:
    print('Create crop parameter dictionary!')

# Efficient Read Videos

for dir in glob.glob('/media/ricky/My Book/Rabbit Research Videos/WP 3.2/C*'):
    camera_text = dir.rsplit('/', 1)[1]
    camera_number = camera_text.rsplit(' ', 1)[1]
    if int(camera_number) < 10:
        vid_file_path = '/media/ricky/My Book/Rabbit Research Videos/WP 3.2/' + camera_text + '/kon0' + camera_number + '.20210630'
    else:
        vid_file_path = '/media/ricky/My Book/Rabbit Research Videos/WP 3.2/' + camera_text + '/kon' + camera_number + '.20210630'

    temp_vid_arr = heat_map_module.read_videos_efficient([vid_file_path + '_050000.mp4', vid_file_path + '_060000.mp4'],
                                                         sample_frame_rate = 7,
                                                         stop_frame= None,
                                                         crop_parameters = crop_parameters[camera_text])

    heat_map_module.numpy_io('write', '/media/ricky/My Book/Rabbit Research Videos/WP 3.2/Analysis/Camera_' + camera_number + '_night_arr.npy', temp_vid_arr)

## Checkpoint Complete! ##
