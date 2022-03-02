# Import statements

from os.path import exists
import glob
import cv2
import json

# Create a crop parameter dictionary

if exists("/media/nipek/My Book/Rabbit Research Videos/WP 3.2/Analysis/Rabbit_Over_Platform/rabbit_over_platform_parameters.json"):
    with open("/media/nipek/My Book/Rabbit Research Videos/WP 3.2/Analysis/Rabbit_Over_Platform/rabbit_over_platform_parameters.json", "r") as f:
        rabbit_over_platform_parameters = json.load(f)
else:
    with open("/media/nipek/My Book/Rabbit Research Videos/WP 3.2/Analysis/Rabbit_Over_Platform/rabbit_over_platform_parameters.json", "w") as f:
        json.dump({"Camera 1": [1, 2], "Camera 2": [1, 2]}, f, indent=4)

# Get crop parameter inspection images (Void)

for cropped_image in glob.glob('/media/nipek/My Book/Rabbit Research Videos/WP 3.2/Analysis/*cropped.jpg'):
    camera_text = cropped_image.rsplit('/', 1)[1].rsplit('_', 1)[0].replace('_', ' ')
    cropped_img = cv2.imread(cropped_image)
    platform_img = cropped_img[rabbit_over_platform_parameters[camera_text][0]:rabbit_over_platform_parameters[camera_text][1], :]
    save_location = '/media/nipek/My Book/Rabbit Research Videos/WP 3.2/Analysis/Rabbit_Over_Platform/Rabbit_Over_Platform_Crop_Inspection/' + \
                    camera_text.replace(' ', '_') + '_platform.jpg'
    print(save_location)
    cv2.imwrite(save_location, platform_img)

## Checkpoint Complete! ##