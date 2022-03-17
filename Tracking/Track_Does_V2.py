# Import Libraries
import os.path
import cv2
import sys
from IndividualTracker import RabbitTracker

# Global parameters
camera_no = 11
action_no = 18
debug = True

# File locations

json_dump_file_location = "Prediction_Dumps/Cam_" + str(camera_no) + "_Action_" + str(action_no) + "_classified.json"
vid_in_file_location = "Action_Videos/Cam_" + str(camera_no) + "_Action_" + str(action_no) + ".mp4"
vid_out_file_location = "Tracked_Videos/Tracker_I_custom_Cam_" + str(camera_no) + "_Action_" + str(action_no) + ".mp4"
output_image = False

# Track rabbits using Euclidian distance of the centroids

rabbit_tracker = RabbitTracker(json_dump_file_location, n_obj=4)
doe0 = rabbit_tracker.initialize_track(0, 0.98)
doe1 = rabbit_tracker.initialize_track(1, 0.98)
doe2 = rabbit_tracker.initialize_track(2, 0.98)
doe3 = rabbit_tracker.initialize_track(3, 0.98)

if debug:
    temp_dict = rabbit_tracker.get_track(3).get_point_dict()
    for key in temp_dict:
        if temp_dict[key]['point'] is not None and 1 < int(key) < 1000:
            #print(key, ':', temp_dict[key])
            continue

#rabbit_tracker.fill_in_between_classified_points(0)
#rabbit_tracker.fill_in_between_classified_points(1)
#rabbit_tracker.fill_in_between_classified_points(2)
rabbit_tracker.fill_in_between_classified_points(3)



track_dict0 = rabbit_tracker.get_track(0).get_point_dict()
track_dict1 = rabbit_tracker.get_track(1).get_point_dict()
track_dict2 = rabbit_tracker.get_track(2).get_point_dict()
track_dict3 = rabbit_tracker.get_track(3).get_point_dict()

# //rabbit_tracker.fill_initial_points()
rabbit_tracker.fill_final_points()
#rabbit_tracker.find_candidate_track(717, 910, (370, 177), (107, 244),  3)
from pprint import pprint
track_c = rabbit_tracker.find_candidate_track(449, 908, (234, 279), (230, 259),  3)
pprint(track_c)
track_c.reverse()
sys.exit(0)

print('----------------------------------------')
if debug:
    temp_dict = rabbit_tracker.get_track(3).get_point_dict()
    for key in temp_dict:
        if temp_dict[key]['point'] is not None and 1 < int(key) < 1000:
            #print(key, ':', temp_dict[key])
            continue



# Visualize the tracked points in the video

# Visualization Function

def custom_visualizer(im, frame_n):
    copy_im = im.copy()
    output_dict = rabbit_tracker.dict[str(frame_n-1)]
    c = 0
    classification_c = 0

    for predictedDoe in output_dict['bbox']:
        cropped_doe = copy_im[predictedDoe[1]:predictedDoe[1] + predictedDoe[3],
                              predictedDoe[0]:predictedDoe[0] + predictedDoe[2]]
        im = cv2.rectangle(im,
                           (predictedDoe[0], predictedDoe[1]),
                           (predictedDoe[0]+predictedDoe[2], predictedDoe[1]+predictedDoe[3]),
                           color=(125, 125, 255),
                           thickness=2)

        im = cv2.putText(im,
                         'Class ' +
                         str(output_dict['classification'][classification_c][0]) + ' ' +
                         str(round(output_dict['classification'][classification_c][1], 4)),
                         (int(predictedDoe[0] + predictedDoe[2] - 65), int(predictedDoe[1]+ predictedDoe[3] - 10)),
                         cv2.FONT_HERSHEY_COMPLEX_SMALL,
                         0.35,
                         (0, 0, 0),
                         0,
                         cv2.LINE_AA)
        classification_c += 1

        for track_id in range(4):
            if rabbit_tracker.get_track(track_id)[frame_n][0] == \
                    rabbit_tracker.get_centroid_bbox(output_dict['bbox'])[c]:
                im = cv2.putText(im,
                                 'Doe ' + str(track_id) + ' Class: ' + str(rabbit_tracker.get_track_doe_id(track_id)),
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
                    cv2.imwrite(os.path.join(folder_path, str(track_id),
                                             str(camera_no) + "_" + str(action_no) + "_" + str(frame_n).zfill(4) + '.jpg'),
                                cropped_doe)
        c += 1
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
        #frame = custom_visualizer(frame, counter)
        frame = cv2.putText(frame, "Frame: " + str(counter), (200, 50), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 0), 3)
        if track_dict0[str(counter)]['point'] is not None:
            frame = cv2.circle(frame, (round(track_dict0[str(counter)]['point'][0]),
                                       round(track_dict0[str(counter)]['point'][1])),
                               3, (255, 0, 255), 3)
            frame = cv2.putText(frame, "Doe 0 - Circle",
                                (round(track_dict0[str(counter)]['point'][0]),
                                 round(track_dict0[str(counter)]['point'][1])-5),
                                cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                0.35,
                                (0, 0, 0),
                                0,
                                cv2.LINE_AA)
        if track_dict1[str(counter)]['point'] is not None:
            frame = cv2.circle(frame, (round(track_dict1[str(counter)]['point'][0]),
                                       round(track_dict1[str(counter)]['point'][1])),
                               3, (255, 125, 0), 3)
            frame = cv2.putText(frame, "Doe 1 - Tail",
                                (round(track_dict1[str(counter)]['point'][0]),
                                 round(track_dict1[str(counter)]['point'][1])-5),
                                cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                0.35,
                                (0, 0, 0),
                                0,
                                cv2.LINE_AA)
        if track_dict2[str(counter)]['point'] is not None:
            frame = cv2.circle(frame, (round(track_dict2[str(counter)]['point'][0]),
                                       round(track_dict2[str(counter)]['point'][1])),
                               3, (0, 125, 255), 3)
            frame = cv2.putText(frame, "Doe 2 - Line",
                                (round(track_dict2[str(counter)]['point'][0]),
                                 round(track_dict2[str(counter)]['point'][1])-5),
                                cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                0.35,
                                (0, 0, 0),
                                0,
                                cv2.LINE_AA)
        if track_dict3[str(counter)]['point'] is not None:
            frame = cv2.circle(frame, (round(track_dict3[str(counter)]['point'][0]),
                                       round(track_dict3[str(counter)]['point'][1])),
                               3, (125, 255, 125), 3)
            frame = cv2.putText(frame, "Doe 3 - Neck",
                                (round(track_dict3[str(counter)]['point'][0]),
                                 round(track_dict3[str(counter)]['point'][1])-5),
                                cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                0.35,
                                (0, 0, 0),
                                0,
                                cv2.LINE_AA)
        video_out.write(frame)
    # Verbose
    if counter % 10 == 0:
        print(counter, " frames of video processed!")
    counter += 1

# Release captions and return Boolean
cap.release()
video_out.release()

#####################################################################
# Notes

# Filter zeros and surrounding ones - post-process
# Implement 3 certain rabbits and one unknown to the tracks
# Initialize and finalize functions
# Solve 2 rabbits assigned to the same path
# Liesbeth methods and Excel files
#####################################################################

