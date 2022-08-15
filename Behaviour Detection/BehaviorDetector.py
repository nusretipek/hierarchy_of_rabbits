from scipy.spatial import distance
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re


class DoeTimeline:

    # init function
    def __init__(self, doe_id, doe_dict):
        self.doe_id = doe_id
        self.doe_dict = doe_dict

        self.position = []
        self.confidence = []
        for key in self.doe_dict:
            if self.doe_dict[key]['point'] is None:
                self.position.append((-1, -1))
                self.confidence.append(-1)

            else:
                self.position.append((round(self.doe_dict[key]['point'][0]), round(self.doe_dict[key]['point'][1])))
                self.confidence.append(self.doe_dict[key]['confidence'])

        self.movement = []
        self.calc_movement()
        self.movement = self.movement_filter(self.movement, min_threshold=5, max_threshold=20)

        self.acceleration = []
        for idx, element in enumerate(self.movement):
            if idx != 0:
                self.acceleration.append(element - self.movement[idx - 1])

        self.active_moments = []
        self.delta_movement = []

    # get functions
    def get_point(self, frame):
        return self.doe_dict[str(frame)]['point']

    def get_confidence(self, frame):
        return self.doe_dict[str(frame)]['confidence']

    def get_position(self, frame):
        return self.position[frame]

    def get_movement(self, frame):
        return self.position[frame]

    # calculate movement
    def calc_movement(self):
        for idx, pos in enumerate(self.position):
            if idx == 0 or self.confidence[idx] <= 0:
                self.movement.append(0)
            else:
                chebyshev_dist = distance.chebyshev(self.position[idx - 1], pos)
                self.movement.append(chebyshev_dist)

    # filter functions
    @staticmethod
    def movement_filter(arr, min_threshold, max_threshold):
        for idx, mov in enumerate(arr):
            if mov > max_threshold or mov < min_threshold:
                arr[idx] = 0
        return arr

    # aggregation to seconds
    def aggregate_movement(self, factor=25):
        agg_arr = []
        temp_avg = 0
        for idx, mov in enumerate(self.movement):
            temp_avg += mov
            if idx != 0 and idx % factor == 0:
                agg_arr.append(temp_avg / factor)
                temp_avg = 0
        return agg_arr

    # plot functions
    def plot_movement(self):
        x = np.arange(0, len(self.movement))
        plt.title("Movement Chart of Doe " + str(self.doe_id))
        plt.xlabel("Frame #")
        plt.ylabel("Movement based on Chebyshev Distance")
        plt.plot(x, self.movement, color="green")
        plt.show()

    def plot_aggregate_movement(self):
        y = self.aggregate_movement(factor=25)
        y = self.movement_filter(y, min_threshold=3, max_threshold=20)
        print(y)
        x = np.arange(0, len(y))
        plt.title("Movement Chart of Doe " + str(self.doe_id))
        plt.xlabel("Frame #")
        plt.ylabel("Movement based on Chebyshev Distance")
        plt.plot(x, y, color="green")
        plt.show()


class BehaviorDetector:

    # init function
    def __init__(self, tracks, n_obj):
        self.n_obj = n_obj
        self.tracks = tracks

        self.timelines = []
        for idx in range(self.n_obj):
            self.timelines.append(DoeTimeline(idx, self.tracks[str(idx)]))

    # get active intervals
    def get_active_intervals(self, fill_in_c):

        # Parameters
        factor = 10
        min_threshold = 5
        max_threshold = 20

        for idx in range(self.n_obj):
            # Aggregate & filter the timeline
            y = self.timelines[idx].aggregate_movement(factor=factor)
            y = self.timelines[idx].movement_filter(y, min_threshold=min_threshold, max_threshold=max_threshold)

            # Get the filtered movement indices
            active_interval_arr = []
            for idy, mov in enumerate(y):
                if mov > 0:
                    active_interval_arr.append(idy)

            # Combine the active intervals
            active_arr = []
            flag = True
            for idy, idz in enumerate(active_interval_arr):
                if flag:
                    active_arr.append([idz, None])
                if idy < len(active_interval_arr) - 1 and \
                        active_interval_arr[idy + 1] - active_interval_arr[idy] <= 1 + fill_in_c:
                    flag = False
                else:
                    active_arr[-1][1] = idz
                    flag = True

            # Convert to frame counts
            active_arr = np.array(active_arr)
            active_arr *= factor
            if len(active_arr) > 0:
                active_arr[:, 1] += factor

            # Assign active_moments attribute
            self.timelines[idx].active_moments = active_arr

    # (Mostly static) Methods to get the closest object, approaching, object and running object
    @staticmethod
    def get_closest_point(p1, other_points):
        min_distance = None
        min_index = None
        for idx, p_x in enumerate(other_points):
            temp_distance = distance.euclidean(p1, p_x)
            if min_index is None or min_distance > temp_distance:
                min_distance = temp_distance
                min_index = idx
        return min_index, min_distance

    @staticmethod
    def get_second_closest_point(p1, other_points):
        distance_list = []
        for idx, p_x in enumerate(other_points):
            temp_distance = distance.euclidean(p1, p_x)
            distance_list.append(temp_distance)
        distance_list = sorted(distance_list)
        return distance_list[1]

    @staticmethod
    def get_approaching_object(p1, previous_positions, current_positions):
        max_distance = None
        max_index = None
        for idx, p_x in enumerate(previous_positions):
            temp_distance_initial = distance.euclidean(p1, p_x)
            temp_distance_final = distance.euclidean(p1, current_positions[idx])
            temp_distance = temp_distance_initial - temp_distance_final
            if max_index is None or max_distance < temp_distance:
                max_distance = temp_distance
                max_index = idx
        return max_index, max_distance

    @staticmethod
    def get_running_object(p1, other_points, other_points2):
        max_distance = None
        max_index = None
        for idx, p_x in enumerate(other_points):
            temp_distance_initial = distance.euclidean(p1, p_x)
            temp_distance_final = distance.euclidean(p1, other_points2[idx])
            temp_distance = temp_distance_final - temp_distance_initial
            if max_index is None or max_distance < temp_distance:
                max_distance = temp_distance
                max_index = idx
        return max_index, max_distance

    def check_target_acceleration(self, doe_id, interval):
        acceleration_c_list = list(map(abs, self.timelines[doe_id].acceleration[interval[0] + 1:interval[1]]))
        acceleration_c = len([x for x in acceleration_c_list if x <= 1]) / (interval[1] - interval[0])
        return acceleration_c

    def check_perpetrator_acceleration(self, doe_id, interval):
        movement_c_list = [1 if x > 0 else 0 for x in self.timelines[doe_id].movement[interval[0] + 1:interval[1]]]
        # print(movement_c_list) len(set(movement_c_list)) == 1 or \

        if len(set(self.timelines[doe_id].movement[interval[0] + 1:interval[1]])) <= 3:
            return 1
        else:
            acceleration_c_list = list(map(abs, self.timelines[doe_id].acceleration[interval[0] + 1:interval[1]]))
            acceleration_c_list = [acceleration_c_list[idx] for idx in range(len(acceleration_c_list))
                                   if movement_c_list[idx] == 1]
            acceleration_c = len([x for x in acceleration_c_list if x <= 1]) / (interval[1] - interval[0])
        return acceleration_c

    @staticmethod
    def longestRepeatedSubstring(s):
        res = ''
        res_length = 0
        index = 0
        n = len(s)
        lcs_re = [[0 for x in range(n + 1)] for y in range(n + 1)]

        for i in range(1, n + 1):
            for j in range(i + 1, n + 1):
                if s[i - 1] == s[j - 1] and lcs_re[i - 1][j - 1] < (j - i):
                    lcs_re[i][j] = lcs_re[i - 1][j - 1] + 1
                    if lcs_re[i][j] > res_length:
                        res_length = lcs_re[i][j]
                        index = max(i, index)
                else:
                    lcs_re[i][j] = 0

        if res_length > 0:
            for i in range(index - res_length + 1, index + 1):
                res = res + s[i - 1]

        return res

    def check_perpetrator_acceleration_position(self, doe_id, interval):
        temp_diff_list = ''
        for idx, pos in enumerate(self.timelines[doe_id].position[interval[0] - 10:interval[1]]):
            if idx > 0:
                temp_diff = distance.euclidean(self.timelines[doe_id].position[interval[0] - 10 + idx - 1], pos)
                temp_diff_list += str(int(temp_diff))
        longest_seq = self.longestRepeatedSubstring(temp_diff_list)
        flag = True
        for idx, element in enumerate(longest_seq):
            if int(element) != 0:
                flag = False
                break
        if not flag:
            p = re.compile(longest_seq)
            indices = [x.start() for x in re.finditer(p, temp_diff_list)]
            if indices[1] - indices[0] != len(longest_seq):
                flag = True
        return flag

    # Get activity indicators using the methods of activity

    # Use static methods of activity to get activity indicators
    def get_activity_indicators(self, doe_id, interval, verbose=True):

        # Get other tracks
        other_tracks = list(range(self.n_obj))
        other_tracks.pop(doe_id)

        # Get position information for self and other tracks
        curr_position_list = self.timelines[doe_id].position
        curr_p0 = curr_position_list[interval[0]]
        curr_p1 = curr_position_list[interval[1]]
        other_initial_points = [self.timelines[idy].position[interval[0]] for idy in other_tracks]
        other_final_points = [self.timelines[idy].position[interval[1]] for idy in other_tracks]
        other_m10_points = [self.timelines[idy].position[interval[0] - 10] for idy in other_tracks]
        other_p10_points = [self.timelines[idy].position[interval[0] + 10] for idy in other_tracks]
        other_final_p10_points = [self.timelines[idy].position[interval[1] + 10] for idy in other_tracks]

        # Calculate metrics for behaviour classification
        min_idx, min_dist = self.get_closest_point(curr_position_list[interval[0]], other_initial_points)
        pre_idx, max_pre_dist = self.get_approaching_object(curr_p0, other_m10_points, other_initial_points)
        post_idx, max_post_dist = self.get_approaching_object(curr_p0, other_initial_points, other_p10_points)
        run_idx, max_run_dist = self.get_running_object(curr_p0, other_initial_points, other_p10_points)
        post_run_idx, post_max_run_dist = self.get_running_object(curr_p1, other_final_points, other_final_p10_points)
        acceleration_c = self.check_target_acceleration(doe_id, interval)
        third_dist = self.get_second_closest_point(curr_position_list[interval[0]], other_initial_points) - min_dist

        # If verbose, print the information
        if verbose:
            print('--------------', '\n',
                  'Doe: ', doe_id, '\n',
                  'Frames:', interval, '\n',
                  'Closest Animal: ', other_tracks[min_idx], min_dist, '\n',
                  'Pre-Approaching Animal: ', other_tracks[pre_idx], max_pre_dist, '\n',
                  'Post-Approaching Animal: ', other_tracks[post_idx], max_post_dist, '\n',
                  'Running Animal: ', other_tracks[run_idx], max_run_dist, '\n',
                  'Acceleration: ', acceleration_c, '\n',
                  'Third_Distance: ', third_dist, '\n',
                  'Post-Run', other_tracks[post_run_idx], post_max_run_dist, '\n',
                  '--------------', '\n')

        # Create return dictionary
        results = {
            'min_c_id': other_tracks[min_idx],
            'min_c_dist': min_dist,
            'pre_app_id': other_tracks[pre_idx],
            'pre_app_dist': max_pre_dist,
            'post_app_id': other_tracks[post_idx],
            'post_app_dist': max_post_dist,
            'pre_p10_run_id': other_tracks[run_idx],
            'pre_p10_run_dist': max_run_dist,
            'post_p10_run_id': other_tracks[post_run_idx],
            'post_p10_run_dist': post_max_run_dist,
            'third_part_dist': third_dist,
            'static_acceleration_index': acceleration_c}

        return results

    # Create sets of related activity moments from related activities
    @staticmethod
    def _check_match(moment, moment_arr, factor_s, max_interval):
        match_list = []

        if moment[1] - moment[0] > max_interval:
            for moment_idz in moment_arr:
                if (moment[0] - factor_s <= moment_idz[0] <= moment[0] + factor_s) or \
                        (moment[0] - factor_s <= moment_idz[1] <= moment[0] + factor_s):
                    match_list.append(moment_idz)

        else:
            for moment_idz in moment_arr:
                if (moment[0] - factor_s <= moment_idz[0] <= moment[0] + factor_s) or \
                        (moment[1] - factor_s <= moment_idz[1] <= moment[1] + factor_s) or \
                        (moment[0] - factor_s <= moment_idz[1] <= moment[0] + factor_s) or \
                        (moment[1] - factor_s <= moment_idz[0] <= moment[1] + factor_s) or \
                        (moment_idz[0] <= moment[0] <= moment_idz[1]) or \
                        (moment_idz[0] <= moment[1] <= moment_idz[1]):
                    match_list.append(moment_idz)

        return match_list

    # Group the active moments to meaningful episodes for accurate classification
    def aggregate_active_moments(self, factor):
        timelines = []
        for idx in range(self.n_obj):
            timelines.append(self.timelines[idx].active_moments)

        timelines_3d = []
        for idx in range(len(timelines)):
            for idy in range(len(timelines[idx])):
                timelines_3d.append([timelines[idx][idy][0], timelines[idx][idy][1], idx])
        temp_timelines_3d = timelines_3d.copy()

        c = 0
        events = {}
        while len(temp_timelines_3d) > 0:
            result = self._check_match(temp_timelines_3d[0], temp_timelines_3d[1:], factor_s=factor, max_interval=100)
            matches = [temp_timelines_3d[0]]
            temp_timelines_3d.pop(0)
            while len(result) > 0:
                matches.append(result[0])
                for i in result:
                    temp_timelines_3d = [value for value in temp_timelines_3d if value != i]
                for element in self._check_match(result[0], temp_timelines_3d, factor_s=factor, max_interval=100):
                    result.append(element)
                result.pop(0)
            events[str(c)] = matches
            c += 1
        return events

    # 1. Check the run-off animal (post and/or pre approaching)
    @staticmethod
    def check_run_off(p, a, b, r, td):
        if a[1] >= 20:
            return a[0], False
        elif b[1] >= 20:
            return b[0], False
        elif r[1] < 10 and a[1] + b[1] > 20:
            return (a[0], False) if (a[1] > b[1]) else (b[0], False)
        elif r[1] < 5 and p[1] < 50:
            return p[0], True
        elif r[1] < 5 and p[1] < 75 and a[1] + b[1] > 10 and ((p[0] == a[0]) or (p[0] == b[0])):
            return p[0], True
        elif r[1] < 5 and p[1] < 80 and (p[0] == a[0] == b[0]):
            return p[0], True
        elif r[1] < 5 and a[1] + b[1] > 15 and (a[0] == b[0] == p[0]):
            return p[0], True
        elif r[1] < 5 and p[1] < 65 and (p[0] == a[0] == b[0]):
            return p[0], True
        elif p[1] < 100 and td > 100:
            return p[0], True
        else:
            return -1, False

    # 2. Check running away animal from the event
    @staticmethod
    def check_perpetrate(r):
        if r[1] >= 10:
            return r[0]
        else:
            return -1

    @staticmethod
    def check_positional_run_off(p, a, b, r):
        if a[1] >= 20 or b[1] >= 20 or (r[1] < 10 and a[1] + b[1] > 20) or \
                (r[1] < 5 and a[1] + b[1] > 15 and (a[0] == b[0] == p[0])):
            return False
        else:
            return True

    # 3. Check displacement of animals during the interval
    def get_displacement(self, doe_idx, interval):
        return min([distance.chebyshev(self.timelines[doe_idx].position[interval[0] + i],
                                       self.timelines[doe_idx].position[interval[1] - i]) for i in range(3)])

    # 4. Check the tracks are overlapping during the action
    def check_track_overlap(self, doe_idx, doe_idy, interval):
        pos_doe_idx = self.timelines[doe_idx].position[interval[0]:interval[1]]
        pos_doe_idy = self.timelines[doe_idy].position[interval[0]:interval[1]]
        avg_dist = 0
        for idx in range(len(pos_doe_idx)):
            avg_dist += (distance.chebyshev(pos_doe_idx[idx], pos_doe_idy[idx]) / len(pos_doe_idx))
        return avg_dist

    # Manipulation for duplicate and circular relationships

    # 1. Start vs End absolute difference < 50 and event is duplicate and/or circular then delete it
    @staticmethod
    def check_dataframe_circular(df, action_start, action_tuple, n_frames=50):
        temp_df = df[abs((action_start - df['Interval End'])).between(0, n_frames)]
        if action_tuple in zip(temp_df.Target, df.Perpetrator):
            return False
        else:
            return True

    # 2. Classify events within the set using activity indicators
    @staticmethod
    def check_cycle(graph):
        def has_cycles(n, s):
            if n in s:
                yield False, n
            else:
                yield True, n
                yield from [i for a, b in graph for i in has_cycles(b, s + [n]) if a == n]

        return not all(a for a, _ in has_cycles(graph[0][0], []))

    # Classify chasing activity (curr_doe -> other doe)
    def classify_chasing(self, chaser_id, activity, event_list, interval_list, run_tuple):

        # Calculate chasing indicators and safety measures to avoid false positives
        temp_tuple = (activity[2], chaser_id)
        other_acc = self.check_perpetrator_acceleration(temp_tuple[1], activity[0:2])
        track_overlap = self.check_track_overlap(temp_tuple[0], temp_tuple[1], activity[0:2])
        displacement = self.get_displacement(activity[2], activity[0:2])

        # Add event to the temporary list
        if temp_tuple not in event_list and other_acc < 0.8 and track_overlap > 50 and displacement > 10:
            event_list.append(temp_tuple)
            interval_list.append(activity[0:2])

            # Check cycles (Dyadic)
            if self.check_cycle(event_list):
                if (temp_tuple[1], temp_tuple[0]) in event_list:
                    idx = event_list.index((temp_tuple[1], temp_tuple[0]))
                    if run_tuple[0] > run_tuple[1][idx]:
                        event_list.pop(idx)
                    else:
                        event_list.pop(-1)
                        interval_list.pop(-1)
                else:
                    event_list.pop(-1)
                    interval_list.pop(-1)

        # Update end time for continuous duplicate events
        if temp_tuple in event_list:
            interval_list[event_list.index(temp_tuple)][1] = activity[1]

        # Return statements
        return event_list, interval_list

    # Classify chasing activity (curr_doe -> other doe)
    def classify_running_off(self, chaser_id, activity, event_list, interval_list, pos_bool):

        # Calculate chasing indicators and safety measures to avoid false positives
        temp_tuple = (chaser_id, activity[2])
        displacement_run_off = self.get_displacement(activity[2], activity[0:2])
        track_overlap = self.check_track_overlap(temp_tuple[0], temp_tuple[1], activity[0:2])

        if not pos_bool:
            other_acc_idx = self.check_perpetrator_acceleration(temp_tuple[0], activity[0:2])
            if other_acc_idx < 0.8:
                other_acc = True
            else:
                other_acc = False
        else:
            temp_movement = self.timelines[chaser_id].movement[activity[0]:activity[1]]
            ratio_movement = sum(temp_movement) / len(temp_movement)
            if ratio_movement < 1.5:
                other_acc = False
            else:
                other_acc = self.check_perpetrator_acceleration_position(temp_tuple[0], activity[0:2])

        # Add event to the temporary list
        if temp_tuple not in event_list and other_acc and track_overlap > 50 and displacement_run_off > 20:
            event_list.append(temp_tuple)
            interval_list.append(activity[0:2])

            # Check cycles (Dyadic)
            if self.check_cycle(event_list):
                event_list.pop(-1)
                interval_list.pop(-1)

        # Update end time for continuous duplicate events
        if temp_tuple in event_list:
            interval_list[event_list.index(temp_tuple)][1] = activity[1]

        # Return statements
        return event_list, interval_list

    @staticmethod
    def find_post_run_set(event_dict, search_frame_id, doe_id):
        for key in event_dict:
            for action in event_dict[key]:
                if action[2] == doe_id and action[0] - 50 < search_frame_id < action[1] + 50:
                    return True
        return False

    # Check running animal in the action dict
    @staticmethod
    def check_event_dict_running_animal(event_dict, doe_id, interval, factor=100):
        for agg_key in event_dict:
            for event in event_dict[agg_key]:
                if event[2] == doe_id and ((interval[0] - factor <= event[0]) or (interval[1] + factor >= event[1])):
                    return True
        return False

    # Main classification function for behaviors
    def classify_events(self, verbose=True):

        # Create an event dictionary
        event_dict = self.aggregate_active_moments(factor=10)
        sorted_event_dict = {k: v for k, v in sorted(event_dict.items(), key=lambda x: (int(x[1][0][0])))}

        # Create empty dataframe for classified events
        df = pd.DataFrame({'Perpetrator': [], 'Target': [], 'Interval Start': [], 'Interval End': []})

        # Main loop for classification
        for key in sorted_event_dict:

            # Group of events, temporary list of classification
            temp_event_tuples = []
            temp_event_intervals = []
            temp_run = []
            temp_post_run = []

            # Nested loop for individual activities within a group
            for activity in sorted(event_dict[key], key=lambda x: (x[0])):

                # Get activity indicators and useful variables for classification
                set_participants = set([i[2] for i in event_dict[key]])
                temp_activity_indicators = self.get_activity_indicators(activity[2], activity[0:2], verbose=verbose)
                c = (temp_activity_indicators['min_c_id'], temp_activity_indicators['min_c_dist'])
                pre = (temp_activity_indicators['pre_app_id'], temp_activity_indicators['pre_app_dist'])
                post = (temp_activity_indicators['post_app_id'], temp_activity_indicators['post_app_dist'])
                run = (temp_activity_indicators['pre_p10_run_id'], temp_activity_indicators['pre_p10_run_dist'])
                post_run = (temp_activity_indicators['post_p10_run_id'], temp_activity_indicators['post_p10_run_dist'])
                third_dist = temp_activity_indicators['third_part_dist']
                accuracy_idx = temp_activity_indicators['static_acceleration_index']

                # Predict behaviour based on activity indicators
                run_off_id, pos_bool = self.check_run_off(c, pre, post, run, third_dist)
                chaser_id = self.check_perpetrate(run)
                post_chaser_id = self.check_perpetrate(post_run)
                temp_run.append(run[1])
                temp_post_run.append(post_run[1])
                run_tuple = (run[1], temp_run)
                post_run_tuple = (post_run[1], temp_post_run)
                flag_chaser = True

                # 1. Chasing Curr_doe -> Other_doe (Pre P10 run)
                run_off_event_bool = self.check_event_dict_running_animal(sorted_event_dict,
                                                                          chaser_id,
                                                                          activity[0:2],
                                                                          factor=100)
                if (chaser_id in set_participants or c[0] == chaser_id) and chaser_id != -1 and \
                        accuracy_idx < 0.7 and run_off_event_bool:
                    temp_len = len(temp_event_tuples)
                    temp_event_tuples, temp_event_intervals = self.classify_chasing(chaser_id,
                                                                                    activity,
                                                                                    temp_event_tuples,
                                                                                    temp_event_intervals,
                                                                                    run_tuple)
                    if len(temp_event_tuples) > temp_len:
                        flag_chaser = False

                # 2. Chasing Curr_doe -> Other_doe (Post P10 run)
                post_run_off_event_bool = self.check_event_dict_running_animal(sorted_event_dict,
                                                                               post_chaser_id,
                                                                               activity[0:2],
                                                                               factor=100)
                participant_bool = (self.find_post_run_set(event_dict, activity[1], post_chaser_id) or
                                    post_chaser_id in set_participants or c[0] == post_chaser_id)
                if participant_bool and post_chaser_id != -1 and accuracy_idx < 0.7 and \
                        (pre[1] + post[1] < post_run[1] or post_run[1] > 25) and post_run_off_event_bool:
                    temp_len = len(temp_event_tuples)
                    temp_event_tuples, temp_event_intervals = self.classify_chasing(post_chaser_id,
                                                                                    activity,
                                                                                    temp_event_tuples,
                                                                                    temp_event_intervals,
                                                                                    post_run_tuple)
                    if len(temp_event_tuples) > temp_len:
                        flag_chaser = False
                # 3. Running off Other_doe -> Curr_doe
                participant_bool = (run_off_id in set_participants or c[0] == run_off_id or pre[0] == post[0])
                if (run_off_id != -1 or pre[1] + post[1] > post_run[1] + run[1] > 0) and accuracy_idx < 0.7 and \
                        ((chaser_id == -1 and post_chaser_id == -1) or flag_chaser):
                    flag_run_off = True
                    if run_off_id == -1:
                        if pre[1] > post[1]:
                            run_off_id = pre[0]
                        else:
                            run_off_id = post[0]
                        chaser_event_bool = self.check_event_dict_running_animal(sorted_event_dict,
                                                                                 run_off_id,
                                                                                 activity[0:2],
                                                                                 factor=100)
                        if not chaser_event_bool:
                            flag_run_off = False

                    if flag_run_off:
                        temp_event_tuples, temp_event_intervals = self.classify_running_off(run_off_id,
                                                                                            activity,
                                                                                            temp_event_tuples,
                                                                                            temp_event_intervals,
                                                                                            pos_bool)

            # Check the duplicate of events and/or cycles in the master dataframe
            for idx in range(len(temp_event_tuples)):
                if verbose:
                    print('Dataframe is checking: ', temp_event_tuples[idx], temp_event_intervals[idx])
                df_c_boolean = self.check_dataframe_circular(df,
                                                             temp_event_intervals[idx][0],
                                                             (temp_event_tuples[idx][0], temp_event_tuples[idx][1]))
                df_d_boolean = self.check_dataframe_circular(df,
                                                             temp_event_intervals[idx][0],
                                                             (temp_event_tuples[idx][1], temp_event_tuples[idx][0]))
                if df_c_boolean and df_d_boolean:
                    df.loc[df.shape[0]] = [temp_event_tuples[idx][0], temp_event_tuples[idx][1],
                                           temp_event_intervals[idx][0], temp_event_intervals[idx][1]]

        # Return master dataframe
        return df.sort_values(by=['Interval Start', 'Interval End'], ignore_index=True)

    # Helper functions for behaviour exploration

    def print_active_moments(self):
        for idx in range(self.n_obj):
            print('Computing for Doe (idx):', idx)
            print(self.timelines[idx].active_moments)

    def check_other_active_moments(self):
        for idx in range(self.n_obj):
            print('Computing for Doe (idx):', idx)
            active_idx = self.timelines[idx].active_moments
            other_tracks = list(range(self.n_obj))
            other_tracks.pop(idx)
            print('Active Moments:', active_idx)
            for idy in other_tracks:
                print('Computing for Doe (idy):', idy)
                active_idy = self.timelines[idy].active_moments
                for active_moment_idx in active_idx:
                    other_initial_points = [self.timelines[idy].position[active_moment_idx[0]] for idy in other_tracks]
                    other_initial_points2 = [self.timelines[idy].position[active_moment_idx[0] + 10] for idy in
                                             other_tracks]
                    min_idx3, min_dist3 = self.get_approaching_object(
                        self.timelines[idx].position[active_moment_idx[0]],
                        other_initial_points, other_initial_points2)
                    for active_moment_idy in active_idy:
                        if abs(active_moment_idx[0] - active_moment_idy[0]) <= 20 and \
                                active_moment_idy[0] - active_moment_idx[1] < 0:
                            print('Active Moment: ', active_moment_idx, ' - Possible Prior:', idy, active_moment_idy)
                            print('Approaching Object: ', other_tracks[min_idx3], min_dist3)
                print('------------------')
            print('%%%%%%%%%%%%%%%%%%%%%%')

    # Plot approximate trajectories
    def get_approximate_trajectory(self, verbose=True, plot=False):
        color_arr = ['#ff00ff', '#007dff', '#ff7d00', '#7dff7d']
        for idx in range(self.n_obj):
            other_tracks = list(range(self.n_obj))
            other_tracks.pop(idx)
            for active_moment in self.timelines[idx].active_moments:
                print(idx, active_moment)
                other_initial_points = [self.timelines[idy].position[active_moment[0]] for idy in other_tracks]
                min_idx, min_dist = self.get_closest_point(self.timelines[idx].position[active_moment[0]],
                                                           other_initial_points)
                other_initial_points2 = [self.timelines[idy].position[active_moment[0] + 10] for idy in other_tracks]
                min_idx2, min_dist2 = self.get_closest_point(self.timelines[idx].position[active_moment[0]],
                                                             other_initial_points2)
                min_idx3, min_dist3 = self.get_approaching_object(self.timelines[idx].position[active_moment[0]],
                                                                  other_initial_points, other_initial_points2)
                if verbose:
                    print('Animal: ', idx, '\n',
                          'Frames:', active_moment, '\n',
                          'Closest Animal: ', other_tracks[min_idx], other_tracks[min_idx2], '\n',
                          'Minimum Distance: ', min_dist, min_dist2, '\n',
                          'Approaching Object: ', other_tracks[min_idx3], min_dist3)

                if plot:
                    x = [i[0] for i in self.timelines[idx].position[active_moment[0]:active_moment[1]]]
                    y = [i[1] for i in self.timelines[idx].position[active_moment[0]:active_moment[1]]]
                    y = np.array(y)
                    y = 480 - y
                    plt.plot(x, y, color=color_arr[idx])
                    for idy in other_tracks:
                        x = [i[0] for i in self.timelines[idy].position[active_moment[0]:active_moment[1]]]
                        y = [i[1] for i in self.timelines[idy].position[active_moment[0]:active_moment[1]]]
                        y = np.array(y)
                        y = 480 - y
                        plt.plot(x, y, color=color_arr[idy])
                    plt.show()
