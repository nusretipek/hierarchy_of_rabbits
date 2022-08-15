import pandas as pd


def cross_ethogram_check(cam_no, sheet_name, start_hour, interval, special_time=None, day_2_offset=None):
    # Parse functions

    def get_relative_time(time_s):
        anchor_time = pd.to_datetime('10-08-2021 ' + start_hour + ':00:00', format='%d-%m-%Y %H:%M:%S')
        if ' ' in time_s:
            return (anchor_time +
                    pd.Timedelta(days=int(time_s[8:10])) +
                    pd.Timedelta(hours=int(time_s[11:13])) +
                    pd.Timedelta(minutes=int(time_s[14:16])) +
                    pd.Timedelta(seconds=int(time_s[17:19])))
        else:
            return (anchor_time +
                    pd.Timedelta(hours=int(time_s[0:2])) +
                    pd.Timedelta(minutes=int(time_s[3:5])) +
                    pd.Timedelta(seconds=int(time_s[6:8])))

    def get_special_relative_time(time_s):
        anchor_time = pd.to_datetime('10-08-2021 ' + special_time['hour'] + ':'
                                     + special_time['minute'] + ':'
                                     + special_time['second'], format='%d-%m-%Y %H:%M:%S')
        if ' ' in time_s:
            return (anchor_time +
                    pd.Timedelta(days=int(time_s[8:10])) +
                    pd.Timedelta(hours=int(time_s[11:13])) +
                    pd.Timedelta(minutes=int(time_s[14:16])) +
                    pd.Timedelta(seconds=int(time_s[17:19])))
        else:
            return (anchor_time +
                    pd.Timedelta(hours=int(time_s[0:2])) +
                    pd.Timedelta(minutes=int(time_s[3:5])) +
                    pd.Timedelta(seconds=int(time_s[6:8])))

    def get_day2_offset(time_s, h):
        return time_s + pd.to_timedelta(int(h), unit='h')

    def convert_doe_id(internal_id):
        if internal_id == 0:
            return 'Doe C'
        elif internal_id == 1:
            return 'Doe A'
        elif internal_id == 2:
            return 'Doe R'
        elif internal_id == 3:
            return 'Doe V'
        else:
            raise AttributeError('Wrong doe id!')

    def filter_eth_df(df, start_time, end_time):
        temp_df = df.copy()
        temp_df = temp_df[temp_df['timestamp_corrected'] >= (start_time - pd.Timedelta(minutes=interval))]
        temp_df = temp_df[temp_df['timestamp_corrected'] <= (end_time + pd.Timedelta(minutes=interval))]
        return temp_df[['Subject', 'Behavior', 'Target_doe']]

    def get_hms_time(col1, col2):
        return pd.to_datetime((col1.rsplit('.', 1)[1][:11] + col2), format='%Y%m%d_%H%M:%S')

    eth_df = pd.read_excel(open('wp3.2 c2.xlsx', 'rb'), sheet_name=sheet_name, engine="openpyxl", dtype=str)
    act_df = pd.read_csv('E:/Rabbit Research Videos/WP32_Cycle2/Action_Videos/Behaviour_Files/Cam_' +
                         str(cam_no) + '.csv', header=0)
    uur_df = pd.read_excel('Behaviour_Files/Camera ' + str(cam_no) + '.xlsx', header=0)
    eth_df = eth_df[eth_df['Subject'] != 'time'].reset_index(drop=True)
    temp_eth_time = eth_df[eth_df['Behavior'].isin(['Agressive action', 'Chasing', 'Fleeing'])].reset_index(drop=True)
    last_record_time = temp_eth_time.iloc[-1]['Time_Relative_hms']
    uur_df['t_stamp'] = uur_df.apply(lambda x: get_hms_time(x['Video_Name'], x['Action_Start']), axis=1)
    a_len = len(
        act_df[act_df['action_no'] <= len(uur_df[uur_df['t_stamp'] <= get_relative_time(last_record_time)])]) + 1

    if special_time is None:
        eth_df['timestamp_corrected'] = eth_df['Time_Relative_hms'].apply(get_relative_time)
    else:
        eth_df['timestamp_corrected'] = eth_df['Time_Relative_hms']
        eth_df.loc[:special_time['row_id'], 'timestamp_corrected'] = eth_df.loc[:special_time['row_id'], 'Time_Relative_hms'].apply(get_special_relative_time)
        eth_df.loc[special_time['row_id']:, 'timestamp_corrected'] = eth_df.loc[special_time['row_id']:, 'Time_Relative_hms'].apply(get_relative_time)

    if day_2_offset is not None:
        for d in day_2_offset:
            eth_df.loc[d['row_id']:, 'timestamp_corrected'] = eth_df.loc[d['row_id']:, 'timestamp_corrected'].apply(
                get_day2_offset, h=d['hour'])

        #eth_special = eth_df.iloc[:special_time['row_id']]
        #eth_normal = eth_df.iloc[special_time['row_id']:]
        #eth_special['timestamp_corrected'] = eth_df['Time_Relative_hms'].apply(get_special_relative_time)
        #eth_normal['timestamp_corrected'] = eth_df['Time_Relative_hms'].apply(get_relative_time)
        #eth_df = pd.concat([eth_special, eth_normal]).reset_index(drop=True)

    eth_df['Target_doe'] = eth_df[['Modifier_4',
                                   'Modifier_5',
                                   'Modifier_6',
                                   'Modifier_7']].apply(lambda x: x.dropna().astype(str).tolist(), axis=1)

    day_1 = eth_df[eth_df['timestamp_corrected'] >= pd.to_datetime('10-08-2021 13:00:00', format='%d-%m-%Y %H:%M:%S')]
    day_1 = day_1[day_1['timestamp_corrected'] <= pd.to_datetime('10-08-2021 20:00:00', format='%d-%m-%Y %H:%M:%S')]
    day_2 = eth_df[eth_df['timestamp_corrected'] >= pd.to_datetime('11-08-2021 08:00:00', format='%d-%m-%Y %H:%M:%S')]
    #print(len(day_1), len(day_2))
    light_df = pd.concat([day_1, day_2], axis=0)

    precision_count = 0
    index_arr = light_df.index.values.tolist()
    index_arr_len = len(index_arr)

    for index, row in act_df.iterrows():
        action_information = uur_df.iloc[row['action_no'] - 1]
        action_start_timestamp = pd.to_datetime(((action_information['Video_Name'].rsplit('.', 1)[1][:11]) +
                                                 action_information['Action_Start']),
                                                format='%Y%m%d_%H%M:%S') + pd.Timedelta(
            seconds=row['Interval Start'] / 25)
        action_end_timestamp = pd.to_datetime(((action_information['Video_Name'].rsplit('.', 1)[1][:11]) +
                                               action_information['Action_Start']),
                                              format='%Y%m%d_%H%M:%S') + pd.Timedelta(seconds=row['Interval End'] / 25)

        initiator_doe = convert_doe_id(row['Perpetrator'])
        receiver_doe = convert_doe_id(row['Target'])

        # Filter eth_df
        filtered_frame = filter_eth_df(eth_df, action_start_timestamp, action_end_timestamp)
        found_flag = False
        for temp_idx, temp_row in filtered_frame.iterrows():
            if receiver_doe in temp_row['Target_doe'] and initiator_doe == temp_row['Subject'] and \
                    temp_row['Behavior'] in ['Agressive action', 'Chasing']:
                found_flag = True
                if temp_idx in index_arr:
                    index_arr.remove(temp_idx)
            if initiator_doe in temp_row['Target_doe'] and receiver_doe == temp_row['Subject'] and \
                    temp_row['Behavior'] in ['Fleeing']:
                found_flag = True
                if temp_idx in index_arr:
                    index_arr.remove(temp_idx)

        if found_flag:
            precision_count += 1
    return round(precision_count / a_len, 4), round((index_arr_len - len(index_arr)) / index_arr_len, 4)


cam_sheet_list = [
    (1, '2-5 c2 eerste uur nog', 12, None),
    (21, '7-10 c2 eerste uur nog', 12, None),
    (22, '12-15 c2 eerste uur nog', 12, None),
    (23, '17-20 c2', 13, None),
    (8, '37-40 c2 ', 13, None),
    (9, '42-45 c2', 13, None),
    (10, '47-50 c2', 13, {'row_id': 230, 'hour': '13', 'minute': '44', 'second': '32'}),
    (12, '57-60 c2', 13, {'row_id': 144, 'hour': '13', 'minute': '58', 'second': '08'}),
    (17, '82-85 c2', 14, None),
    (18, '87-90 c2', 14, None),
    (19, '92-95 c2', 14, None),
    (20, '97-100 c2', 14, None)]

cam_sheet_list = [
    (1, '2-5 c2 eerste uur nog', 12, None, None),
    (21, '7-10 c2 eerste uur nog', 12, None, [{'row_id': 680, 'hour': '1'}, {'row_id': 736, 'hour': '1'}]),
    (22, '12-15 c2 eerste uur nog', 12, None, None),
    (23, '17-20 c2', 13, None, [{'row_id': 306, 'hour': '1'}]),
    (8, '37-40 c2 ', 13, None, [{'row_id': 909, 'hour': '1'}, {'row_id': 967, 'hour': '1'}, {'row_id': 980, 'hour': '1'}]),
    (9, '42-45 c2', 13, None, None),
    (10, '47-50 c2', 13, {'row_id': 230, 'hour': '13', 'minute': '44', 'second': '32'}, [{'row_id': 757, 'hour': '1'}]),
    (12, '57-60 c2', 13, {'row_id': 144, 'hour': '13', 'minute': '58', 'second': '08'}, [{'row_id': 1013, 'hour': '1'}]),
    (17, '82-85 c2', 14, None, [{'row_id': 452, 'hour': '2'}]),
    (18, '87-90 c2', 14, None, [{'row_id': 299, 'hour': '1'}]),
    (19, '92-95 c2', 14, None, [{'row_id': 840, 'hour': '1'}]),
    (20, '97-100 c2', 14, None, None)]

# p, r = cross_ethogram_check(19, '92-95 c2', '14', 0.5)
# print(p, r)
# sys.exit(0)

def cross_ethogram_check_2(cam_no, sheet_name):
    eth_df = pd.read_excel(open('wp3.2 c2.xlsx', 'rb'), sheet_name=sheet_name, engine="openpyxl", dtype=str)
    eth_df = eth_df[eth_df['Subject'] != 'time'].reset_index(drop=True)
    print(cam_no,
          len(eth_df[eth_df['Behavior'] == 'Agressive action']),
          len(eth_df[eth_df['Behavior'] == 'Chasing']),
          len(eth_df[eth_df['Behavior'] == 'Fleeing']))

for (cam, sheet, s_time, special_time_dict, day2_offset_dict) in cam_sheet_list:
    cross_ethogram_check_2(cam, sheet)

sys.exit(0)
for (cam, sheet, s_time, special_time_dict, day2_offset_dict) in cam_sheet_list:
    s = str(cam)
    temp_arr = []
    for i in [1, 5, 30]:
        p, r = cross_ethogram_check(cam, sheet, str(s_time), i, special_time_dict, day2_offset_dict)
        temp_arr.append((p, r, i))

    for (p, r, i) in temp_arr:
        s += ' - ' + str(i) + '[Precision: ' + str(p) + ', Recall: ' + str(r) + ']'
    print(s)
