#Import Packages
import subprocess
def install(name):
    subprocess.call(['pip', 'install', name])

## Basics
try:
  import cupy as cp
except:
  install('cupy-cuda111')
  print("Cupy is installing!")
finally:
  import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
import gdown
import math
import random
from scipy import stats

## Computer Vision

import cv2

## Utilities

import os
import glob
import time

def fetch_video_from_gdrive(id, name):
  url = 'https://drive.google.com/u/0/uc?id=' + id + '&export=download'
  gdown.download(url, name, quiet=False)

def read_single_video_loose(vid, sample_frame_count=15, grayscale=False):
  start_time = time.time()
  cap = cv2.VideoCapture(vid)

  ### Create empty Numpy Array 4D with UINT8 type
  if grayscale:
    video_np = np.empty(
      (math.ceil(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / sample_frame_count), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
       int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))), np.dtype('uint8'))
  else:
    video_np = np.empty(
      (math.ceil(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / sample_frame_count), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
       int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 3), np.dtype('uint8'))

  ### Create counters
  counter, counter_frames = 0, 0

  ### Loop the frames and save them in array
  success = True
  while success:
    success, image = cap.read()
    if success:
      if counter % sample_frame_count == 0:
        if grayscale:
          image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        video_np[counter_frames] = image
        counter_frames += 1
      counter += 1

  print("Total execution time for extraction of frames (Loose):", round((time.time() - start_time)), "seconds")
  return video_np


def get_video_loose_flat(arr):
  start_time = time.time()

  ### Create 1D array of Numpy arrays across the frames sequentially
  video_np_flat = np.empty((arr.shape[1]*arr.shape[2]), dtype=object)

  ### Create a counter to iterate
  counter = 0

  ### Loop columns and rows to save across frames
  for col_wise in range(arr.shape[1]):
    for row_wise in range(arr.shape[2]):
      video_np_flat[counter] = arr[:, col_wise, row_wise]
      counter += 1

  print("Total execution time for Numpy flattening:" , (time.time()-start_time), "seconds")
  return video_np_flat


def read_multiple_videos_efficient(vid_array, use_gpu = False, sample_frame_rate = None):
  start_time = time.time()
  cap = cv2.VideoCapture(vid_array[0])

  if use_gpu:

    ### Create empty Numpy Array 4D with UINT8 type
    video_cp = cp.zeros((int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) * int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 256),
                        dtype='uint32')

    for vid in vid_array:
      cap = cv2.VideoCapture(vid)

      if sample_frame_rate is not None:
        random_frames = cp.random.randint(1, sample_frame_rate, size=(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))
      else:
        random_frames = cp.ones((int(cap.get(cv2.CAP_PROP_FRAME_COUNT))), )

      ### Create counter
      counter = 0

      ### Loop the frames and save them in array
      success = True
      while success:
        success, image = cap.read()
        if success and random_frames[counter] == 1:
          video_cp[cp.arange(int(video_cp.shape[0])), cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).flatten()] += 1
        counter += 1
        if counter % 10000 == 0:
          print('Video:', vid, '- Frame:', counter)

    # Return statements
    print("Total execution time for extraction of frames (RAM efficient - GPU):", round((time.time() - start_time)),
          "seconds")
    return video_cp

  else:

    ### Create empty Numpy Array 4D with UINT8 type
    video_np = np.zeros((int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) * int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 256),
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
          video_np[np.arange(int(video_np.shape[0])), cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).flatten()] += 1
        counter += 1
        if counter % 10000 == 0:
          print('Video:', vid, '- Frame:', counter)

    # Return statements
    print("Total execution time for extraction of frames (RAM efficient - No GPU):", round((time.time() - start_time)),
          "seconds")
    return video_np

def get_kde_efficient(arr, bandwidth, gridsize, random_sample_size):
  start_time = time.time()
  local_maxima_threshold = 1e-04

  ### Create empty frames for local maxima and standard deviation
  frame_local_maxima = np.empty((arr.shape[0], arr.shape[1]), np.dtype('uint8'))
  frame_std = np.empty((arr.shape[0], arr.shape[1]), np.dtype('float64'))

  ### Iterate each pixel to calculate density values

  for row in range(arr.shape[0]):
    for column in range(arr.shape[1]):

      ### Calculate and differentiate Gaussian KDE
      repeat_np = np.repeat(np.arange(0, arr.shape[2]), arr[row, column].tolist())
      g_kde = stats.gaussian_kde(np.random.choice(repeat_np, random_sample_size), bw_method=bandwidth)
      g_kde_values = g_kde(np.linspace(0, 256, gridsize))
      local_maxima_points = (np.diff(np.sign(np.diff(g_kde_values))) < 0).nonzero()[0] + 1
      local_maxima_points_len = len(local_maxima_points)

      ### Eliminate small local maxima points
      for local_maxima in local_maxima_points:
        if g_kde_values[local_maxima] < local_maxima_threshold:
          local_maxima_points_len -= 1

      frame_local_maxima[row, column], frame_std[row, column] = local_maxima_points_len, np.std(repeat_np)

  ### Verbose
  print("Total execution time for pixel KDE:", (time.time() - start_time), "seconds")
  ### Return statement
  return frame_local_maxima, frame_std

def get_mask_cage_type1(mask_type, std_arr=None, kde_arr=None, tier1_threshold=25, lower_threshold=15):
  if mask_type == 'std' and std_arr is not None:
    frame_std_temp = std_arr.copy()
    frame_std_temp[frame_std_temp > lower_threshold] = 255
    frame_std_temp[frame_std_temp <= lower_threshold] = 0
    print("Zero pixel count: ", len(np.where(frame_std_temp == 0)[0]))
    return frame_std_temp

  elif mask_type == 'kde' and kde_arr is not None:
    frame_local_maxima_temp = kde_arr.copy()
    frame_local_maxima_temp[frame_local_maxima_temp > 1] = 255
    frame_local_maxima_temp[frame_local_maxima_temp <= 1] = 0
    print("Zero pixel count: ", len(np.where(frame_local_maxima_temp == 0)[0]))
    return frame_local_maxima_temp

  elif mask_type == 'combined' and kde_arr is not None and std_arr is not None:

    combined_mask = np.zeros((480, 720), np.dtype('uint8'))
    for row in range(combined_mask.shape[0]):
      for column in range(combined_mask.shape[1]):
        if (std_arr[row, column] > tier1_threshold or kde_arr[row, column] > 2 or (
                kde_arr[row, column] > 1 and std_arr[row, column] > lower_threshold)):
          combined_mask[row, column] = 255
    print("Zero pixel count: ", len(np.where(combined_mask == 0)[0]))
    return combined_mask

  else:
    print('Please enter one of the following valid mask types: [std, kde, combined]')
    return None

def get_background_image(arr, mask):
  start_time = time.time()
  background_image = np.full((int(arr.shape[0]), int(arr.shape[1])), 255, np.dtype('uint8'))

  for row in range(arr.shape[0]):
    for column in range(arr.shape[1]):
      if mask is not None and mask[row, column] == 0:
        background_image[row, column] = arr[row, column].argmax()
      if mask is None:
        background_image[row, column] = arr[row, column].argmax()

  ### Return statements
  print("Total execution time for background image:", (time.time()-start_time), "seconds")
  return background_image


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

def get_peak_std(arr, bandwidth='silverman', adj_constant=30, random_sample_size=2000):
  start_time = time.time()

  ### Create empty frames for local maxima and standard deviation
  frame_peak_std = np.empty((arr.shape[0], arr.shape[1]), np.dtype('float64'))

  ### Iterate each pixel to calculate density values

  for row in range(arr.shape[0]):
    for column in range(arr.shape[1]):
      ### Calculate and differentiate Gaussian KDE
      repeat_np = np.repeat(np.arange(0, arr.shape[2]), arr[row, column].tolist())
      g_kde = stats.gaussian_kde(np.random.choice(repeat_np, random_sample_size), bw_method=bandwidth)
      g_kde_values = g_kde(
        np.linspace(arr[row, column].argmax() - adj_constant, arr[row, column].argmax() + adj_constant, 10))
      frame_peak_std[row, column] = np.std(g_kde_values)

  ### Verbose
  print("Total execution time for KDE peak standard deviation:", (time.time() - start_time), "seconds")

  ### Return statement
  return frame_peak_std

def normalize_peak_std(arr):
  temp_arr = arr.copy()
  temp_arr[temp_arr > 0.03] = 0.03
  normalized_temp_arr = ((temp_arr - np.min(temp_arr)) / (np.max(temp_arr) - np.min(temp_arr)))
  return normalized_temp_arr

def normalize_peak_curvature(arr):
  temp_arr = arr.copy()
  temp_arr = np.abs(temp_arr)
  temp_arr[temp_arr > np.median(temp_arr)] = np.median(temp_arr)
  normalized_temp_arr = (temp_arr - np.min(temp_arr)) / (np.max(temp_arr) - np.min(temp_arr))
  return normalized_temp_arr

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

def get_heat_map_cage_type1(vid, mask, row_low=50, row_high=400, col_low=100, col_high=630):
  start_time = time.time()
  cap = cv2.VideoCapture(vid)

  ## Create a 2D array of type uint32
  frame_heat_map = np.zeros((row_high - row_low, col_high - col_low), np.dtype('uint32'))

  ## Create kernels
  kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

  ## Reading video frame by frame
  success = True
  while success:
    success, image = cap.read()
    if success:
      ### convert each image to grayscale
      gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)[row_low:row_high, col_low:col_high]
      thresh, bw_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
      ### Apply heatmap gaussian kernel
      opening = cv2.morphologyEx(bw_img, cv2.MORPH_OPEN, kernel)
      ### Multiply with the mask values (make sure the type is uint8)
      masked_image = opening * (mask[row_low:row_high, col_low:col_high] / 255).astype('uint8')
      ### Adjust the new created array
      frame_heat_map += (masked_image / 255).astype('uint32')

  ##return to the matrix
  print("Total execution time for heat map:", (time.time() - start_time), "seconds")
  return frame_heat_map / int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

def normalize_heat_map(arr):
  temp_arr = arr.copy()
  return (temp_arr - np.min(temp_arr)) / (np.max(temp_arr) - np.min(temp_arr))

def blend_heatmap(background_image, heat_map, threshold=None):
  # Create RGB and alpha channels
  foreground = np.full((heat_map.shape[0], heat_map.shape[1], 3), (0, 0, 255), np.dtype('uint8')).astype(float)
  background = cv2.cvtColor(cv2.imread(background_image, cv2.IMREAD_GRAYSCALE), cv2.COLOR_GRAY2RGB).astype(float)
  alpha = normalize_heat_map(heat_map).astype(float)

  # Remove errors
  if threshold is not None:
    alpha[alpha > threshold] = 0

  # Broadcast the alpha channel to third dimension
  alpha = np.stack((alpha, alpha, alpha), -1)

  # Blend with alpha channel
  foreground = cv2.multiply(alpha, foreground)
  background = cv2.multiply(1.0 - alpha, background)
  out_image = cv2.add(foreground, background)

  # Return statement
  return out_image

def read_videos_efficient_cage_type2(vid_array, use_gpu = False, sample_frame_rate=15, stop_frame=None, crop_parameters=None):
  start_time = time.time()
  cap = cv2.VideoCapture(vid_array[0])

  if use_gpu:
    ### Create empty Numpy Array 2D with UINT32 type
    if crop_parameters is None:
      video_np = cp.zeros((int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) * int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 256),
                          dtype='uint32')
    else:
      video_np = cp.zeros(((crop_parameters[1] - crop_parameters[0]) * (crop_parameters[3] - crop_parameters[2]), 256),
                          dtype='uint32')
  else:
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
    if use_gpu:
      ### Loop the frames and save them in array
      success = True
      while success:
        success, image = cap.read()
        if success and random_frames[counter] == 1:
          if crop_parameters is None:
            cv2.imwrite('frame' + str(counter) + '.jpg', image)
            video_np[cp.arange(int(video_np.shape[0])), cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).flatten()] += 1
          else:
            video_np[cp.arange(int(video_np.shape[0])),
                     cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)[crop_parameters[0]:crop_parameters[1],
                     crop_parameters[2]:crop_parameters[3]].flatten()] += 1
        counter += 1
        if stop_frame is not None and counter > stop_frame:
          success = False
        if counter % 10000 == 0:
          print('Video:', vid, '- Frame:', counter)
    else:
      ### Loop the frames and save them in array
      success = True
      while success:
        success, image = cap.read()
        if success and random_frames[counter] == 1:
          if crop_parameters is None:
            cv2.imwrite('frame' + str(counter) + '.jpg', image)
            video_np[np.arange(int(video_np.shape[0])), cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).flatten()] += 1
          else:
            video_np[np.arange(int(video_np.shape[0])),
                     cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)[crop_parameters[0]:crop_parameters[1],
                     crop_parameters[2]:crop_parameters[3]].flatten()] += 1
        counter += 1
        if stop_frame is not None and counter > stop_frame:
          success = False
        if counter % 10000 == 0:
          print('Video:', vid, '- Frame:', counter)

    # Return statements
    if use_gpu:
      print("Total execution time for extraction of frames (RAM efficient - GPU):", round((time.time() - start_time)),
          "seconds")
      video_np = cp.asnumpy(video_np)
    else:
      print("Total execution time for extraction of frames (RAM efficient - No GPU):", round((time.time() - start_time)),
          "seconds")
    return video_np.reshape(crop_parameters[1] - crop_parameters[0], crop_parameters[3] - crop_parameters[2], 256)

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

def get_mask_cage_type2(arr, frame_std, peak):
  mask = np.full((arr.shape[0], arr.shape[1]), 255, dtype = 'uint8')
  peak = np.abs(peak)
  arr = arr.argmax(axis=2)
  mask[np.logical_and(np.logical_and((arr > 130), (frame_std <= 25)), (peak > 0.01))] = 0
  mask[np.logical_and(np.logical_or((peak < 0.035), (frame_std > 20)), (arr <= 160))] = 255
  return mask

def get_heat_map_cage_type2(vid, mask, crop_parameters):
  start_time = time.time()
  cap = cv2.VideoCapture(vid)

  ## Create a 2D array of type uint32
  frame_heat_map = np.zeros(((crop_parameters[1] - crop_parameters[0]), (crop_parameters[3] - crop_parameters[2])),
                            np.dtype('uint32'))

  ## Reading video frame by frame
  success = True
  while success:
    success, image = cap.read()
    if success:
      ### convert each image to grayscale
      gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)[crop_parameters[0]:crop_parameters[1],
                 crop_parameters[2]:crop_parameters[3]]
      thresh, bw_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

      ### Multiply with the mask values (make sure the type is uint8)
      masked_image = bw_img * (mask / 255).astype('uint8')

      ### Adjust the new created array
      frame_heat_map += (masked_image / 255).astype('uint32')

  ##return to the matrix
  print("Total execution time for heat map:", (time.time() - start_time), "seconds")
  return frame_heat_map / int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

def get_heat_map_interval_cage_type2(vid, mask, crop_parameters, start_frame, stop_frame):
  start_time = time.time()
  cap = cv2.VideoCapture(vid)

  ## Create a 2D array of type uint32
  frame_heat_map = np.zeros(((crop_parameters[1] - crop_parameters[0]), (crop_parameters[3] - crop_parameters[2])),
                            np.dtype('uint32'))

  ## Reading video frame by frame
  counter = 0
  success = True
  while success:
    success, image = cap.read()
    if success and (counter >= start_frame):
      ### convert each image to grayscale
      gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)[crop_parameters[0]:crop_parameters[1],
                 crop_parameters[2]:crop_parameters[3]]
      thresh, bw_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

      ### Multiply with the mask values (make sure the type is uint8)
      masked_image = bw_img * (mask / 255).astype('uint8')

      ### Adjust the new created array
      frame_heat_map += (masked_image / 255).astype('uint32')

    if counter == stop_frame:
      success = False
    counter += 1

  ##return to the matrix
  print("Total execution time for heat map:", (time.time() - start_time), "seconds")
  return frame_heat_map