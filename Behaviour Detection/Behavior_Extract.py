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

# INITIAL TESTS
# action_no = str(4)
# action_no = str(9)
# action_no = str(10)
# action_no = str(10) + '_c12'
# action_no = str(13)
# action_no = str(15)
# action_no = str(16)
# action_no = str(19)
# action_no = str(50)
# action_no = str(52)
# action_no = str(66)

# SECOND TESTS
# action_no = str(9) + '_teta'
# action_no = str(15) + '_teta'
# action_no = str(16) + '_teta'
# action_no = str(19) + '_beta'
# action_no = str(21) + '_teta'
# action_no = str(24) + '_beta'
# action_no = str(34) + '_beta'
# action_no = str(40) + '_teta'
# action_no = str(71) + '_beta'
# action_no = str(102) + '_beta'

# PRESENTATION TESTS
action_no = str(18) + '_c11'
# action_no = str(420)

# File locations

json_dump_file_location = "C:\\Users\\nipek\\Desktop\\Cycle 1 Analysis\\reid_track\\Action_008_classified.json" #"../Tracking/Action_Videos/random_action_videos/Action_" + str(action_no) + "_classified.json"
print(json_dump_file_location)

# json_dump_file_location = "../Tracking/Prediction_Dumps/Cam_" + str(camera_no) + "_Action_" + str(action_no) + "_classified.json"


output_image = False

# Track rabbits using Classification Model & Euclidian distance of the centroids
rabbit_tracker = RabbitTracker(json_dump_file_location, n_obj=4)
rabbit_tracker.generate_tracks(threshold=0.98)

tracks = dict()
for track_id in range(rabbit_tracker.n_obj):
    tracks[str(track_id)] = rabbit_tracker.get_track(track_id).get_point_dict()

obj = BehaviorDetector(tracks, len(tracks))

########## FIX FOR EXTRAPOLATION ###########
# obj.timelines[3].plot_aggregate_movement()
# print(obj.timelines[3].confidence[1900:1999])
# sys.exit(0)
###########################################
# print(obj.timelines[2].acceleration[1100:1130])
# print(obj.timelines[2].movement[1100:1130])


obj.get_active_intervals(fill_in_c=2)
print(obj.aggregate_active_moments(factor=10))
# obj.get_activity_indicators()
# obj.get_approximate_trajectory()
# pprint(obj.aggregate_active_moments(10))
print(obj.classify_events(verbose=True))
sys.exit(0)

"""
for idx, mov in enumerate(DoeTimeline(2, tracks[str(2)]).movement):
    print(idx, mov)
"""
color_arr = ['#FF00FF', '#007DFF', '#FF7D00', '#7DFF7D']
shape_arr = ['Circle', 'Tail', 'Line', 'Neck']
fig, ax = plt.subplots(1, 1, figsize=(15, 10))
ax.set_title("Filtered Movement Chart")
ax.set_xlabel("Frame (t)")
ax.set_ylabel("Aggregated Movement")
for idx, track_id in enumerate(range(rabbit_tracker.n_obj)):
    temp_track_dict = tracks[str(track_id)]
    doe_timeline = DoeTimeline(track_id, temp_track_dict)
    y = doe_timeline.aggregate_movement(factor=10)
    y = doe_timeline.movement_filter(y, min_threshold=5, max_threshold=20)
    x = np.arange(0, len(y)) * 10
    ax.plot(x, y, color=color_arr[idx], label=shape_arr[idx])
plt.legend(loc='upper right')
plt.show()
fig.savefig('filtered_movement.png', format='png', dpi=300)

# print(doe_timeline.movement)
# print(doe_timeline.aggregate_movement())
# doe_timeline.plot_movement()
# doe_timeline.plot_aggregate_movement()


# print(doe_timeline.get_position(0))
# System exit
sys.exit(0)
