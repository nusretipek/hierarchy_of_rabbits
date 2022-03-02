# Import Packages

import cv2
import numpy as np
from scipy import stats
import math
import time
import random
import os
from os.path import exists
from os import walk
import glob
import matplotlib.pyplot as plt

# Mask Functions

## Read videos for mask

def read_videos_efficient(vid_array, sample_frame_rate=15,
                                     stop_frame=None, crop_parameters=None):
  start_time = time.time()
  cap = cv2.VideoCapture(vid_array[0])

 ### Create empty Numpy Array 2D with UINT32 type
  if crop_parameters is None:
    video_np = np.zeros((int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) * int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 256),
                        dtype='uint32')
  else:
    video_np = np.zeros(((crop_parameters[1] - crop_parameters[0]) * (crop_parameters[3] - crop_parameters[2]), 256),
                        dtype='uint32')

  for vid in vid_array:
    cap = cv2.VideoCapture(vid)

    if sample_frame_rate is not None:
      random_frames = np.random.randint(1, sample_frame_rate, size=(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))
    else:
      random_frames = np.ones((int(cap.get(cv2.CAP_PROP_FRAME_COUNT))), )

    ### Create counter
    counter = 0

    ### Loop the frames and save them in array
    success = True
    while success:
      success, image = cap.read()
      if success and random_frames[counter] == 1:
        video_np[np.arange(int(video_np.shape[0])),
                cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)[crop_parameters[0]:crop_parameters[1],
                crop_parameters[2]:crop_parameters[3]].flatten()] += 1
      counter += 1
      if stop_frame is not None and counter > stop_frame:
        success = False
      if counter % 10000 == 0:
        print('Video:', vid, '- Frame:', counter)

  # Return statements
  print("Total execution time for extraction of frames (RAM efficient - No GPU):", round((time.time() - start_time)),
        "seconds")
  return video_np.reshape(crop_parameters[1] - crop_parameters[0], crop_parameters[3] - crop_parameters[2], 256)

## Get standard deviation of pixels for mask

def pixel_std(arr):

  start_time = time.time()

  ### Create empty frames for local maxima and standard deviation
  frame_std = np.empty((arr.shape[0], arr.shape[1]), np.dtype('float64'))

  ### Iterate each pixel to calculate density values

  for row in range(arr.shape[0]):
    for column in range(arr.shape[1]):
      ### Calculate and differentiate Gaussian KDE
      repeat_np = np.repeat(np.arange(0, arr.shape[2]), arr[row, column].tolist())
      frame_std[row, column] = np.std(repeat_np)

  ### Verbose
  print("Total execution time for standard deviation:", (time.time() - start_time), "seconds")

  ### Return statement
  return frame_std

## Get peak sharpness of pixels for mask

def get_peak_sharpness(arr, bandwidth='silverman', adj_constant=30, random_sample_size=2000):
  start_time = time.time()

  ### Create empty frames for local maxima and standard deviation
  frame_peak = np.empty((arr.shape[0], arr.shape[1]), np.dtype('float64'))

  ### Iterate each pixel to calculate density values

  for row in range(arr.shape[0]):
    for column in range(arr.shape[1]):
      ### Calculate and differentiate Gaussian KDE
      repeat_np = np.repeat(np.arange(0, arr.shape[2]), arr[row, column].tolist())
      g_kde = stats.gaussian_kde(np.random.choice(repeat_np, random_sample_size), bw_method=bandwidth)
      g_kde_values = g_kde(np.linspace(arr[row, column].argmax() - adj_constant, arr[row, column].argmax() + adj_constant, 3))
      frame_peak[row, column] = np.diff(np.diff(g_kde_values))[0]

  ### Verbose
  print("Total execution time for KDE peakness:", (time.time() - start_time), "seconds")

  ### Return statement
  return frame_peak

def get_mask_wp3_2(arr, frame_std, peak):
  mask = np.full((arr.shape[0], arr.shape[1]), 255, dtype = 'uint8')
  peak = np.abs(peak)
  arr = arr.argmax(axis=2)
  mask[np.logical_and(np.logical_and((arr > 50), (frame_std <= 35)), (peak > 0.02))] = 0
  mask[np.logical_and(np.logical_or((peak < 0.045), (frame_std > 23)), (arr <= 200))] = 255
  return mask

## Heatmap generation

def get_heat_map_wp2(vid, mask, crop_parameters, vid_name = ''):
  start_time = time.time()
  cap = cv2.VideoCapture(vid)
  ## Frame skip
  frame_skip_constant = 25
  evaluated_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / frame_skip_constant)

  ## Create a 2D array of type uint32
  frame_heat_map = np.zeros(((crop_parameters[1] - crop_parameters[0]), (crop_parameters[3] - crop_parameters[2])),
                            np.dtype('uint32'))

  ## Create kernels
  kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
  in_mask = mask.copy()
  in_mask[in_mask <250] = 0
  in_mask[in_mask >=250] = 1
  mask_255_uint8 = in_mask.astype('uint8')

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
  return frame_heat_map / evaluated_frames

def get_heat_map_wp2_pool(vid, mask, heatmap_path, crop_parameters, vid_name = ''):
  start_time = time.time()
  cap = cv2.VideoCapture(vid)
  ## Frame skip
  frame_skip_constant = 25
  evaluated_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / frame_skip_constant)

  ## Create a 2D array of type uint32
  frame_heat_map = np.zeros(((crop_parameters[1] - crop_parameters[0]), (crop_parameters[3] - crop_parameters[2])),
                            np.dtype('uint32'))

  ## Create kernels
  kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
  in_mask = mask.copy()
  in_mask[in_mask <250] = 0
  in_mask[in_mask >=250] = 1
  mask_255_uint8 = in_mask.astype('uint8')

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

  ## Return Void
  print("Total execution time for heat map:", vid_name, "-", (time.time() - start_time), "seconds")
  numpy_io('write', heatmap_path, (frame_heat_map / evaluated_frames))

## Heatmap normalization

def normalize_heatmap_wp2(arr):
  temp_arr = arr.copy()
  return ((temp_arr - np.min(temp_arr)) / (np.max(temp_arr) - np.min(temp_arr)))

## Get mask for rabbit over the platform

def get_mask_rabbit_over_platform_wp3_2(arr, frame_std, peak):
  mask = np.full((arr.shape[0], arr.shape[1]), 255, dtype = 'uint8')
  peak = np.abs(peak)
  arr = arr.argmax(axis=2)
  mask[np.logical_and((arr > 100), (frame_std <= 25))] = 0
  mask[np.logical_and(np.logical_or((peak < 0.07), (frame_std > 15)), (arr <= 130))] = 255
  return mask

# Utility Functions

## Numpy IO

def numpy_io(operation, filename, arr = None):
  if operation == 'read':
    with open(filename, 'rb') as f:
      return np.load(f)
  elif (operation == 'write') and (arr is not None):
    with open(filename, 'wb') as f:
      np.save(f, arr)
    print(filename, ' - Successfully written!')
  elif (operation == 'write') and (arr is None):
    print('Adjust arr parameter!')
  else:
    print('Specify one of the following correct operations: [read, write]')
    return False

## Get first frame

def get_initial_frame(vid, crop_check = False, crop_params = None):
  cap = cv2.VideoCapture(vid)
  cap.set(1,0)
  success, image = cap.read()
  if crop_check:
    cap = cv2.VideoCapture(vid)
    cap.set(1, 0)
    success, image = cap.read()
    image = image[crop_params[0]:crop_params[1],crop_params[2]:crop_params[3]]
  return image