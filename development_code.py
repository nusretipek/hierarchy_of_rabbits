# Read video files into numpy
# wp_camera_1 = heat_map_module.read_videos_efficient_cage_type2([vid_file],
#                                               sample_frame_rate = 15,
#                                               stop_frame = None,
#                                               crop_parameters = crop_parameters['camera_1'])

# heat_map_module.numpy_io('write', 'camera_1_arr.npy', wp_camera_1)

# camera_1_arr = heat_map_module.numpy_io('read', 'camera_1_arr.npy')
# camera_1_pixel_std = heat_map_module.pixel_std(camera_1_arr)
# camera_1_peak_sharpness = heat_map_module.get_peak_sharpness(camera_1_arr)
# camera_1_mask = heat_map_module.get_mask_wp3_2(camera_1_arr, camera_1_pixel_std, camera_1_peak_sharpness)
# heat_map_module.numpy_io('write', 'camera_1_sharpness.npy', camera_1_peak_sharpness)
# heat_map_module.numpy_io('write', 'camera_1_mask.npy', camera_1_mask)
# cv2.imwrite('camera_1_mask.jpg', camera_1_mask)

import cv2
import heat_map_module
import time
import numpy as np
import glob
import multiprocessing as mp

def get_heat_map_wp2(vid, mask, crop_parameters, vid_name=''):
    start_time = time.time()
    cap = cv2.VideoCapture(vid)
    ## Frame skip
    frame_skip_constant = 100
    evaluated_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / frame_skip_constant)

    ## Create a 2D array of type uint32
    frame_heat_map = np.zeros(((crop_parameters[1] - crop_parameters[0]), (crop_parameters[3] - crop_parameters[2])),
                              np.dtype('uint32'))

    ## Create kernels
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    in_mask = mask.copy()
    mask[mask < 250] = 0
    mask[mask >= 250] = 1
    mask_255_uint8 = mask.astype('uint8')

    ## Reading video frame by frame
    counter = 0
    success = True
    while success:
        cap.set(1, counter * frame_skip_constant)
        success, image = cap.read()
        if success:
            ### Convert each image to grayscale
            gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)[crop_parameters[0]:crop_parameters[1],
                       crop_parameters[2]:crop_parameters[3]]
            thresh, bw_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

            ### Multiply with the mask values (make sure the type is uint8)
            masked_image = bw_img * mask_255_uint8

            ### Adjust the new created array
            frame_heat_map += ((masked_image / 255).astype('uint32'))
        counter += 1

    ## Return to the matrix
    print("Total execution time for heat map:", vid_name, "-", (time.time() - start_time), "seconds")
    print("Eval Frames:", evaluated_frames)
    return frame_heat_map / evaluated_frames


# arr = get_heat_map_wp2('/media/ricky/My Book/Rabbit Research Videos/WP 3.2/Camera 10/kon10.20210629_160000.mp4',
#                       cv2.imread('/media/ricky/My Book/Rabbit Research Videos/WP 3.2/Analysis/Final_Masks/Camera_10_mask.jpg', cv2.IMREAD_GRAYSCALE),
#                       [150, 420, 115, 610],
#                       'Development')

# cv2.imwrite('/media/ricky/My Book/Rabbit Research Videos/WP 3.2/Analysis/heat_map_dev.jpg', arr*255)
# print(arr[7:9,25:45])
# a = cv2.imread('/media/ricky/My Book/Rabbit Research Videos/WP 3.2/Analysis/Final_Masks/Camera_10_mask.jpg', cv2.IMREAD_GRAYSCALE)
# heat_map_module.numpy_io('write', '/media/ricky/My Book/Rabbit Research Videos/WP 3.2/Analysis/dev_arr.npy', arr)
# temp_arr = heat_map_module.numpy_io('read', '/media/ricky/My Book/Rabbit Research Videos/WP 3.2/Analysis/Heatmaps/Camera 10/kon10.20210629_160000.npy')
# print(temp_arr[7:9,25:45])
# print(arr[7:9,25:45])
# import glob
# for directory in glob.glob('/media/ricky/My Book/Rabbit Research Videos/WP 3.2/Camera 1'):
#    print(directory)

def test_cage_open_wp2(heatmap_file):
    img = cv2.imread(heatmap_file, cv2.IMREAD_GRAYSCALE)
    blur = cv2.GaussianBlur(img, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for i in contours:
        area = cv2.contourArea(i)
        if area > 1000:
            thresh = cv2.drawContours(thresh, contours, 0, (0, 255, 0), 3)
    return heatmap_file.rsplit('/', 1)[1].rsplit('.', 1)[0], np.sum(thresh == 255)

def detect_outliers_wp2(data):
    index_list = []
    m = np.mean(data)
    s = np.std(data)
    for index in range(len(data)):
        if not (m - 2 * s < data[index] < m + 2 * s):
            index_list.append(index)
    return index_list

white_pixel_list = []
camera_name_list = []
start = time.time()
for heat_map in glob.glob('/media/nipek/My Book/Rabbit Research Videos/WP 3.2/Analysis/Heatmaps/Camera 10/*.jpg'):
    name, white_pixel = test_cage_open_wp2(heat_map)
    camera_name_list.append(name)
    white_pixel_list.append(white_pixel)
print(time.time() - start, ' SECONDS!')

start = time.time()
with mp.Pool(6) as p:
    pool_result = p.map(test_cage_open_wp2,
                                  glob.glob('/media/nipek/My Book/Rabbit Research Videos/WP 3.2/Analysis/Heatmaps/Camera 10/*.jpg'))
    camera_name_list = [x[0] for x in pool_result]
    white_pixel_list = [x[1] for x in pool_result]
print(time.time() - start, ' SECONDS!')

index_list = detect_outliers_wp2(np.array(white_pixel_list))
print(index_list)
print(np.array(camera_name_list)[index_list])
print([path for path in glob.glob('/media/nipek/My Book/Rabbit Research Videos/WP 3.2/C*') if (('Camera 1' in path) or ('Camera 10' in path))])

for dir in glob.glob('/media/nipek/My Book/Rabbit Research Videos/WP 3.2/C*'):
    camera_text = dir.rsplit('/', 1)[1]
    if camera_text in ['Camera 1', 'Camera 10']:
        continue
    print(camera_text)