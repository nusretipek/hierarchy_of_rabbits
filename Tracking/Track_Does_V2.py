# Import Libraries
from scipy.spatial.distance import euclidean
import cv2
import sys
from IndividualTracker import RabbitTracker

# Global parameters
camera_no = 21  # 11 21
action_no = 19  # 18 19
debug = False

# File locations


# Cycle 1
json_dump_file_location = "C:\\Users\\nipek\\Desktop\\Cycle 1 Analysis\\reid_track\\Action_007_classified.json"
vid_in_file_location = "C:\\Users\\nipek\\Desktop\\Cycle 1 Analysis\\reid_track\\Action_007.mp4"
vid_out_file_location = "C:\\Users\\nipek\\Desktop\\Cycle 1 Analysis\\reid_track\\Track_Cam10_Action_007.mp4"
output_image = False

'''
json_dump_file_location = "Prediction_Dumps/Cam_" + str(camera_no) + "_Action_" + str(action_no) + "_classified.json"
vid_in_file_location = "Action_Videos/Cam_" + str(camera_no) + "_Action_" + str(action_no) + ".mp4"
vid_out_file_location = "Tracked_Videos/Tracker_Final_Cam_" + str(camera_no) + "_Action_" + str(action_no) + ".mp4"
output_image = False
'''
# Track rabbits using Classification Model & Euclidian distance of the centroids
rabbit_tracker = RabbitTracker(json_dump_file_location, n_obj=4)
rabbit_tracker.generate_tracks(threshold=0.98)

tracks = dict()
for track_id in range(rabbit_tracker.n_obj):
    tracks[str(track_id)] = rabbit_tracker.get_track(track_id).get_point_dict()

if debug:
    temp_dict = tracks[str(0)]
    for key in temp_dict:
        if temp_dict[key]['point'] is not None and 2290 < int(key) < 2325:
            print(key, ':', temp_dict[key])
            continue

# Visualize the tracked points in the video

# Visualization Function


def custom_visualizer(im, frame_n):
    output_dict = rabbit_tracker.dict[str(frame_n-1)]
    classification_c = 0
    color_custom_arr = [(255, 0, 255), (255, 125, 0), (0, 125, 255), (125, 255, 125)]

    frame_track_arr = []
    for track_idx in range(rabbit_tracker.n_obj):
        frame_track_arr.append(tracks[str(track_idx)][str(str(frame_n-1))]['point'])

    for predictedDoe in output_dict['bbox']:
        min_idx = None
        min_dist = None
        for idx, track_centroid in enumerate(frame_track_arr):
            temp_dist = euclidean(track_centroid, (int(predictedDoe[0] + predictedDoe[2]/2),
                                                   int(predictedDoe[1] + predictedDoe[3]/2)))
            if min_idx is None or temp_dist < min_dist:
                min_dist = temp_dist
                min_idx = idx

        '''
        cropped_doe = copy_im[predictedDoe[1]:predictedDoe[1] + predictedDoe[3],
                              predictedDoe[0]:predictedDoe[0] + predictedDoe[2]]
        '''

        im = cv2.rectangle(im,
                           (predictedDoe[0], predictedDoe[1]),
                           (predictedDoe[0]+predictedDoe[2], predictedDoe[1]+predictedDoe[3]),
                           color=color_custom_arr[min_idx],
                           thickness=2)
        '''
        im = cv2.putText(im,
                         'Class ' +
                         str(output_dict['classification'][classification_c][0]) + ' ' +
                         str(round(output_dict['classification'][classification_c][1], 4)),
                         (int(predictedDoe[0] + predictedDoe[2] - 65), int(predictedDoe[1] + predictedDoe[3] - 10)),
                         cv2.FONT_HERSHEY_COMPLEX_SMALL,
                         0.35,
                         (0, 0, 0),
                         0,
                         cv2.LINE_AA)
        '''
        classification_c += 1
        '''
        for track_idx in range(4):
            if rabbit_tracker.get_track(track_idx)[frame_n][0] == \
                    rabbit_tracker.get_centroid_bbox(output_dict['bbox'])[c]:
                im = cv2.putText(im,
                                 'Doe ' + str(track_idx) + ' Class: ' + str(rabbit_tracker.get_track_doe_id(track_idx)),
                                 (int(predictedDoe[0] + 5), int(predictedDoe[1]+10)),
                                 cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                 0.35,
                                 (0, 0, 0),
                                 0,
                                 cv2.LINE_AA)
                if output_image:
                    folder_path = 'Doe_Images_Temp/' + str(camera_no) + '_' + str(action_no)
                    if not os.path.exists(folder_path):
                        os.mkdir(folder_path)
                        os.mkdir(os.path.join(folder_path, '0'))
                        os.mkdir(os.path.join(folder_path, '1'))
                        os.mkdir(os.path.join(folder_path, '2'))
                        os.mkdir(os.path.join(folder_path, '3'))
                    cv2.imwrite(os.path.join(folder_path,
                                             str(track_idx),
                                             str(camera_no) + "_" + str(action_no) +
                                             "_" + str(frame_n).zfill(4) + '.jpg'), cropped_doe)
        c += 1
        '''
    return im


# Parameters
cap = cv2.VideoCapture(vid_in_file_location)
counter = 1
video_out = cv2.VideoWriter(vid_out_file_location, cv2.VideoWriter_fourcc(*'mp4v'), 25,
                            (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

# Loop video
success = True
while success:
    cap.set(1, counter)
    success, frame = cap.read()
    if success:
        # To use bounding boxes & image output =>
        #   { frame = custom_visualizer(frame, counter) }

        frame = custom_visualizer(frame, counter)
        # Frame counter on the top for debugging
        #frame = cv2.putText(frame, "Frame: " + str(counter), (250, 50), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 0), 3)

        # Visualizing track circles
        color_arr = [(255, 0, 255), (255, 125, 0), (0, 125, 255), (125, 255, 125)]
        shape_arr = ['Circle', 'Tail', 'Line', 'Neck']

        for track_id in range(rabbit_tracker.n_obj):
            temp_track_dict = tracks[str(track_id)]
            if tracks[str(track_id)][str(counter)]['point'] is not None:
                frame = cv2.circle(frame, (round(tracks[str(track_id)][str(counter)]['point'][0]),
                                           round(tracks[str(track_id)][str(counter)]['point'][1])),
                                   3, color_arr[track_id], 3)

                frame = cv2.putText(frame, shape_arr[track_id],  # "Doe " + str(track_id) + " - " +
                                    (round(tracks[str(track_id)][str(counter)]['point'][0]),
                                     round(tracks[str(track_id)][str(counter)]['point'][1])-5),
                                    cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                    0.5,
                                    (0, 0, 0),
                                    1,
                                    cv2.LINE_AA)
        video_out.write(frame)

    # Verbose
    if counter % 100 == 0:
        print(counter, " frames of video processed!")
    counter += 1

# Release captions and return Boolean
cap.release()
video_out.release()

# System exit
sys.exit(0)
