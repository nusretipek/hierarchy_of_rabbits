# Import statements

import heat_map_module
import numpy as np
import glob
import pandas as pd
pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 5)

# Create aggregated Numpy Arrays

for dataframe in glob.glob('/media/nipek/My Book/Rabbit Research Videos/WP 3.2/Analysis/Master_Heatmaps/Dataframes/*'):
    camera_text = dataframe.rsplit('/', 1)[1].rsplit('.', 1)[0]

    # For daily aggregation
    temp_df = pd.read_csv(dataframe, index_col= None)
    temp_df['date_time'] = pd.to_datetime(temp_df['date_time'])
    unique_list_days = set(map(lambda x: x.day, temp_df['date_time'].to_list()))
    for c_date in unique_list_days:
        filtered_temp_df = temp_df[temp_df['date_time'].dt.day == c_date]
        heat_map_locations = filtered_temp_df['heat_map_location'].to_list()
        frame_counts = filtered_temp_df['frame_count'].to_list()
        counter = 0
        total_frame_count_weight = 0
        daily_heatmap = None
        for heatmap in heat_map_locations:
            temp_heatmap = heat_map_module.numpy_io('read', heatmap)
            if daily_heatmap is None:
                daily_heatmap = np.zeros((temp_heatmap.shape[0],temp_heatmap.shape[1]), dtype = 'float64')
            daily_heatmap += (frame_counts[counter] * temp_heatmap)
            total_frame_count_weight += frame_counts[counter]
            counter += 1
        daily_heatmap /= total_frame_count_weight
        save_location = '/media/nipek/My Book/Rabbit Research Videos/WP 3.2/Analysis/Master_Heatmaps/Daily/' + camera_text + '_day_' + str(c_date) +'.npy'
        heat_map_module.numpy_io('write', save_location, daily_heatmap)

    # For daylight aggregation
    temp_df = pd.read_csv(dataframe, index_col=None)
    temp_df['date_time'] = pd.to_datetime(temp_df['date_time'])
    unique_list_days = set(map(lambda x: x.day, temp_df['date_time'].to_list()))
    day_light_videos = []
    day_light_videos_frame_count = []
    for c_date in unique_list_days:
        if c_date == 29:
            filtered_temp_df = temp_df[temp_df['date_time'].dt.day == c_date]
            heat_map_locations = filtered_temp_df['heat_map_location'].to_list()
            frame_counts = filtered_temp_df['frame_count'].to_list()
            counter = 0
            total_frame_count_weight = 0
            daily_light_heatmap = None
            for heatmap in heat_map_locations:
                temp_heatmap = heat_map_module.numpy_io('read', heatmap)
                if daily_light_heatmap is None:
                    daily_light_heatmap = np.zeros((temp_heatmap.shape[0], temp_heatmap.shape[1]), dtype='float64')
                daily_light_heatmap += (frame_counts[counter] * temp_heatmap)
                total_frame_count_weight += frame_counts[counter]
                counter += 1
            daily_light_heatmap /= total_frame_count_weight
            save_location = '/media/nipek/My Book/Rabbit Research Videos/WP 3.2/Analysis/Master_Heatmaps/Light/' + camera_text + '_day_' + str(
                c_date) + '.npy'
            heat_map_module.numpy_io('write', save_location, daily_light_heatmap)
            day_light_videos += heat_map_locations
            day_light_videos_frame_count += frame_counts
        else:
            filtered_temp_df = temp_df[temp_df['date_time'].dt.day == c_date]
            filtered_temp_df = filtered_temp_df[(filtered_temp_df['date_time'].dt.hour >= 8) & (filtered_temp_df['date_time'].dt.hour < 20)]
            heat_map_locations = filtered_temp_df['heat_map_location'].to_list()
            frame_counts = filtered_temp_df['frame_count'].to_list()
            counter = 0
            total_frame_count_weight = 0
            daily_light_heatmap = None
            for heatmap in heat_map_locations:
                temp_heatmap = heat_map_module.numpy_io('read', heatmap)
                if daily_light_heatmap is None:
                    daily_light_heatmap = np.zeros((temp_heatmap.shape[0], temp_heatmap.shape[1]), dtype='float64')
                daily_light_heatmap += (frame_counts[counter] * temp_heatmap)
                total_frame_count_weight += frame_counts[counter]
                counter += 1
            daily_light_heatmap /= total_frame_count_weight
            save_location = '/media/nipek/My Book/Rabbit Research Videos/WP 3.2/Analysis/Master_Heatmaps/Light/' + camera_text + '_day_' + str(
                c_date) + '.npy'
            heat_map_module.numpy_io('write', save_location, daily_light_heatmap)
            day_light_videos += heat_map_locations
            day_light_videos_frame_count += frame_counts
    counter = 0
    total_frame_count_weight = 0
    master_daily_light_heatmap = None
    for heatmap in day_light_videos:
        temp_heatmap = heat_map_module.numpy_io('read', heatmap)
        if master_daily_light_heatmap is None:
            master_daily_light_heatmap = np.zeros((temp_heatmap.shape[0], temp_heatmap.shape[1]), dtype='float64')
        master_daily_light_heatmap += (day_light_videos_frame_count[counter] * temp_heatmap)
        total_frame_count_weight += day_light_videos_frame_count[counter]
        counter += 1
    master_daily_light_heatmap /= total_frame_count_weight
    save_location = '/media/nipek/My Book/Rabbit Research Videos/WP 3.2/Analysis/Master_Heatmaps/Light/' + camera_text + '_all.npy'
    heat_map_module.numpy_io('write', save_location, master_daily_light_heatmap)

    # For dark aggregation
    temp_df = pd.read_csv(dataframe, index_col=None)
    temp_df['date_time'] = pd.to_datetime(temp_df['date_time'])
    unique_list_days = set(map(lambda x: x.day, temp_df['date_time'].to_list()))
    dark_videos = []
    dark_videos_frame_count = []
    for c_date in unique_list_days:
        if c_date != 29:
            filtered_temp_df = temp_df[temp_df['date_time'].dt.day == c_date]
            filtered_temp_df = filtered_temp_df[(filtered_temp_df['date_time'].dt.hour >= 20) | (filtered_temp_df['date_time'].dt.hour < 8)]
            heat_map_locations = filtered_temp_df['heat_map_location'].to_list()
            frame_counts = filtered_temp_df['frame_count'].to_list()
            counter = 0
            total_frame_count_weight = 0
            daily_dark_heatmap = None
            for heatmap in heat_map_locations:
                temp_heatmap = heat_map_module.numpy_io('read', heatmap)
                if daily_dark_heatmap is None:
                    daily_dark_heatmap = np.zeros((temp_heatmap.shape[0], temp_heatmap.shape[1]), dtype='float64')
                daily_dark_heatmap += (frame_counts[counter] * temp_heatmap)
                total_frame_count_weight += frame_counts[counter]
                counter += 1
            daily_dark_heatmap /= total_frame_count_weight
            save_location = '/media/nipek/My Book/Rabbit Research Videos/WP 3.2/Analysis/Master_Heatmaps/Dark/' + camera_text + '_day_' + str(
                c_date) + '.npy'
            heat_map_module.numpy_io('write', save_location, daily_dark_heatmap)
            dark_videos += heat_map_locations
            dark_videos_frame_count += frame_counts
    counter = 0
    total_frame_count_weight = 0
    master_daily_dark_heatmap = None
    for heatmap in dark_videos:
        temp_heatmap = heat_map_module.numpy_io('read', heatmap)
        if master_daily_dark_heatmap is None:
            master_daily_dark_heatmap = np.zeros((temp_heatmap.shape[0], temp_heatmap.shape[1]), dtype='float64')
        master_daily_dark_heatmap += (dark_videos_frame_count[counter] * temp_heatmap)
        total_frame_count_weight += dark_videos_frame_count[counter]
        counter += 1
    master_daily_dark_heatmap /= total_frame_count_weight
    save_location = '/media/nipek/My Book/Rabbit Research Videos/WP 3.2/Analysis/Master_Heatmaps/Dark/' + camera_text + '_all.npy'
    heat_map_module.numpy_io('write', save_location, master_daily_dark_heatmap)

    # For 72-hour aggregation
    heat_map_locations = temp_df['heat_map_location'].to_list()
    frame_counts = temp_df['frame_count'].to_list()
    counter = 0
    total_frame_count_weight = 0
    master_hour_heatmap = None
    for heatmap in heat_map_locations:
        temp_heatmap = heat_map_module.numpy_io('read', heatmap)
        if master_hour_heatmap is None:
            master_hour_heatmap = np.zeros((temp_heatmap.shape[0], temp_heatmap.shape[1]), dtype='float64')
        master_hour_heatmap += (frame_counts[counter] * temp_heatmap)
        total_frame_count_weight += frame_counts[counter]
        counter += 1
    master_hour_heatmap /= total_frame_count_weight
    save_location = '/media/nipek/My Book/Rabbit Research Videos/WP 3.2/Analysis/Master_Heatmaps/72_Hours/' + camera_text + '.npy'
    heat_map_module.numpy_io('write', save_location, master_hour_heatmap)

## Checkpoint Complete!##
