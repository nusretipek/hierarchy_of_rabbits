# Import Libraries
import os.path
from pprint import pprint

import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.spatial.distance
from scipy.spatial import distance
from BehaviorDetector import DoeTimeline, BehaviorDetector


sys.path.insert(0, '../Tracking')
from IndividualTracker import RabbitTracker

# Global parameters
camera_no = 21  # 11 21
action_no = 9  # 18 19

# File locations

json_dump_file_location = "../Tracking/Action_Videos/random_action_videos/Action_" + str(action_no) + "_classified.json"
vid_in_file_location = "../Tracking/Action_Videos/random_action_videos/Action_" + str(action_no) + ".mp4"
vid_out_file_location = "../Tracking/Action_Videos/random_action_videos/Tracker_Final_Action_" + str(action_no) + ".mp4"
print(json_dump_file_location)

#json_dump_file_location = "../Tracking/Prediction_Dumps/Cam_" + \
#                          str(camera_no) + "_Action_" + str(action_no) + "_classified.json"
#vid_in_file_location = "../Tracking/Action_Videos/Cam_" + \
#                       str(camera_no) + "_Action_" + str(action_no) + ".mp4"
#vid_out_file_location = "../Behaviour Detection/Tracker_Final_Cam_" + \
#                        str(camera_no) + "_Action_" + str(action_no) + ".mp4"

output_image = False

# Track rabbits using Classification Model & Euclidian distance of the centroids
rabbit_tracker = RabbitTracker(json_dump_file_location, n_obj=4)
rabbit_tracker.generate_tracks(threshold=0.98)

tracks = dict()
for track_id in range(rabbit_tracker.n_obj):
    tracks[str(track_id)] = rabbit_tracker.get_track(track_id).get_point_dict()

obj = BehaviorDetector(tracks, len(tracks))

########## FIX FOR EXTRAPOLATION ###########
#obj.timelines[3].plot_aggregate_movement()
#print(obj.timelines[3].confidence[1900:1999])
#sys.exit(0)
###########################################
print(obj.timelines[2].acceleration[1100:1130])
print(obj.timelines[2].movement[1100:1130])

#sys.exit(0)

obj.get_active_intervals(fill_in_c=2)
print(obj.aggregate_active_moments(factor=10))
#obj.get_activity_indicators()
#obj.get_approximate_trajectory()
#pprint(obj.aggregate_active_moments(10))
print(obj.classify_events(verbose=False))

sys.exit(0)

"""
for idx, mov in enumerate(DoeTimeline(2, tracks[str(2)]).movement):
    print(idx, mov)
"""

fig, ax = plt.subplots(2, 2, figsize=(15, 10))
ax = ax.flatten()
for track_id in range(rabbit_tracker.n_obj):
    temp_track_dict = tracks[str(track_id)]
    doe_timeline = DoeTimeline(track_id, temp_track_dict)
    y = doe_timeline.movement

    y = doe_timeline.aggregate_movement(factor=10)
    y = doe_timeline.movement_filter(y, min_threshold=5, max_threshold=20)
    x = np.arange(0, len(y))
    print(ax)
    ax[track_id].set_title("Movement Chart of Doe " + str(track_id))
    ax[track_id].set_xlabel("Frame #")
    ax[track_id].set_ylabel("Aggregated Movement")
    ax[track_id].plot(x, y, color="green")
plt.show()


    #print(doe_timeline.movement)
    #print(doe_timeline.aggregate_movement())
    #doe_timeline.plot_movement()
    #doe_timeline.plot_aggregate_movement()


#print(doe_timeline.get_position(0))
# System exit
sys.exit(0)
