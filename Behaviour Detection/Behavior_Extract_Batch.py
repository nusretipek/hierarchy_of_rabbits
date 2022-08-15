# Import Libraries
import sys
import pandas as pd
import glob
from BehaviorDetector import BehaviorDetector
from IndividualTracker import RabbitTracker

# Global parameters
cam_no = 1
master_df = None

# File locations
#for json_dump_file_location in sorted(glob.glob('E:/Rabbit Research Videos/WP32_Cycle2/Action_Videos/AD_Cam'

for json_dump_file_location in sorted(glob.glob('AD_Cam1' + '/*.json')):
    print(json_dump_file_location)
    action_no = int(json_dump_file_location.rsplit('\\', 1)[1].rsplit('_', 1)[0].rsplit('_', 1)[1])

    try:
        # Tracker
        rabbit_tracker = RabbitTracker(json_dump_file_location, n_obj=4)
        rabbit_tracker.generate_tracks(threshold=0.98)

        tracks = dict()
        for track_id in range(rabbit_tracker.n_obj):
            tracks[str(track_id)] = rabbit_tracker.get_track(track_id).get_point_dict()

        # Behaviour Classifier
        obj = BehaviorDetector(tracks, len(tracks))
        obj.get_active_intervals(fill_in_c=2)
        temp_df = obj.classify_events(verbose=False)
        temp_df['action_no'] = action_no
        if len(temp_df) > 0:
            if master_df is None:
                master_df = temp_df
            else:
                master_df = pd.concat([master_df, temp_df], ignore_index=True)

    except Exception as e:
        print("Exception Video " + str(action_no) + ': ', e)

master_df.to_csv('Behaviour_Files/NMCam_' + str(cam_no) + '.csv', index=False)


