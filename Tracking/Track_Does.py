import Tracker
import cv2


# Track rabbits using Euclidian distance of the centroids

rabbit_tracker = Tracker.RabbitTracker("Cam_11_Action_7.json", n_obj=4)
rabbit_tracker.generate_tracks()
doe0 = rabbit_tracker.get_track(0)
doe1 = rabbit_tracker.get_track(1)
doe2 = rabbit_tracker.get_track(2)
doe3 = rabbit_tracker.get_track(3)


# Visualize the tracked points in the video

# Visualization Function


def custom_visualizer(im, frame_n):
    output_dict = rabbit_tracker.dict[str(frame_n-1)]
    c = 0
    for predictedDoe in output_dict['bbox']:
        im = cv2.rectangle(im,
                           (predictedDoe[0], predictedDoe[1]),
                           (predictedDoe[0]+predictedDoe[2], predictedDoe[1]+predictedDoe[3]),
                           color=(125, 125, 255),
                           thickness=2)
        for track_id in range(4):
            if rabbit_tracker.get_track(track_id)[frame_n] == rabbit_tracker.get_centroid_bbox(output_dict['bbox'])[c]:
                im = cv2.putText(im,
                                 'Doe ' + str(track_id),
                                 (int(predictedDoe[0] + 5), int(predictedDoe[1]+10)),
                                 cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                 0.35,
                                 (0, 0, 0),
                                 0,
                                 cv2.LINE_AA)
        c += 1
    return im


# Parameters
vid = "Cam_11_Action_7.mp4"
cap = cv2.VideoCapture(vid)
counter = 1
video_out = cv2.VideoWriter('Tracker_custom_Cam_11_Action_7.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 25,
                            (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
# Loop video
success = True
while success:
    cap.set(1, counter)
    success, frame = cap.read()
    if success:
        frame = cv2.circle(frame, (doe0[counter][0], doe0[counter][1]), 3, (255, 0, 0), 3)
        frame = cv2.circle(frame, (doe1[counter][0], doe1[counter][1]), 3, (0, 0, 255), 3)
        frame = cv2.circle(frame, (doe2[counter][0], doe2[counter][1]), 3, (0, 255, 255), 3)
        frame = cv2.circle(frame, (doe3[counter][0], doe3[counter][1]), 3, (255, 0, 255), 3)
        frame = custom_visualizer(frame, counter)
        video_out.write(frame)
    # Verbose
    if counter % 10 == 0:
        print(counter, " frames of video processed!")
    counter += 1

# Release captions and return Boolean
cap.release()
video_out.release()
