# Import statements

import os
import glob
import warnings
import cv2
warnings.simplefilter(action='ignore', category=FutureWarning)

# Loop Cameras
for dir in sorted(glob.glob('D:\\Rabbit Research Videos\\WP 3.2\\C*')):
    camera_text = dir.rsplit('\\', 1)[1]
    action_clips_path = os.path.join('D:\\Rabbit Research Videos\\HPC_Analysis\\WP32\\Action_Video_Clips', camera_text)

    skip_frames = 0
    counter = 0
    initial_state = True

    for action_clip in sorted(glob.glob(action_clips_path + '\\action_*.mp4'), key=os.path.getmtime):
        print(action_clip, skip_frames)
        cap = cv2.VideoCapture(action_clip)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if initial_state:
            video = cv2.VideoWriter(os.path.join(action_clips_path, "action_sampled_MP4V_Codec.mp4"),
                                    cv2.VideoWriter_fourcc(*"MP4V"),
                                    fps,
                                    (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
            initial_state = False

        if frame_count > skip_frames:
            cap.set(1, skip_frames)
            skip_frames = 0
        else:
            skip_frames = skip_frames - frame_count
            continue

        video_bool = True
        while video_bool:
            success, frame = cap.read()
            if not success:
                video_bool = False
            if counter > fps*60:
                if frame_count-(skip_frames+(fps*60)) > fps*300:
                    cap.set(1, frame_count-(skip_frames+(fps*60))+fps*300)
                    counter = 0
                else:
                    video_bool = False
                    skip_frames = fps*300 - (frame_count-(skip_frames+(fps*60)))
                    print('Skip: ', skip_frames)
                counter = 0
            video.write(frame)
            counter += 1

    video.release()
    break

## Checkpoint Complete!##
