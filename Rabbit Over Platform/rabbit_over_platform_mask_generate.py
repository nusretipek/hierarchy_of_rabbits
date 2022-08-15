# Import statements

import json
from os.path import exists

import cv2
import numpy as np

# Generate and save masks

if exists("/media/nipek/My Book/Rabbit Research Videos/WP 3.2/Analysis/crop_parameters.json"):
    with open("/media/nipek/My Book/Rabbit Research Videos/WP 3.2/Analysis/crop_parameters.json", "r") as f:
        crop_parameters = json.load(f)
else:
    print('Create crop parameter dictionary!')

if exists(
        "/media/nipek/My Book/Rabbit Research Videos/WP 3.2/Analysis/Rabbit_Over_Platform/rabbit_over_platform_parameters.json"):
    with open(
            "/media/nipek/My Book/Rabbit Research Videos/WP 3.2/Analysis/Rabbit_Over_Platform/rabbit_over_platform_parameters.json",
            "r") as f:
        rabbit_over_platform_parameters = json.load(f)
else:
    print('Create platform parameters dictionary!')


def get_sample_image_dark(vid, minute, second, crop_parameters, platform_parameters):
    cap = cv2.VideoCapture(vid)
    success = True
    while success:
        cap.set(1, 25 * (minute * 60 + second))
        success, image = cap.read()
        if success:
            gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)[crop_parameters[0]:crop_parameters[1],
                                                               crop_parameters[2]:crop_parameters[3]]
            thresh, bw_img = cv2.threshold(gray_img, 230, 255, cv2.THRESH_BINARY_INV)
            bw_img = bw_img[platform_parameters[0]:platform_parameters[1], :]
            kernel = np.ones((5, 5), np.uint8)
            opening = cv2.morphologyEx(bw_img, cv2.MORPH_CLOSE, kernel)
            success = False
    cv2.imshow("Test", opening)
    cv2.waitKey(0)
    return opening


params = [['/media/nipek/My Book/Rabbit Research Videos/WP 3.2/Camera 5/kon05.20210701_030000.mp4', 5, 22]]

mask = get_sample_image_dark('/media/nipek/My Book/Rabbit Research Videos/WP 3.2/Camera 5/kon05.20210701_030000.mp4', 5,
                             22,
                             crop_parameters['Camera 5'], rabbit_over_platform_parameters['Camera 5'])
cv2.imwrite('/media/nipek/My Book/Rabbit Research Videos/WP 3.2/Analysis/Rabbit_Over_Platform/Masks/Camera_5.jpg', mask)
