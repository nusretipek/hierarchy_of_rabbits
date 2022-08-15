from scipy.spatial import distance
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
                self.acceleration.append(element - self.movement[idx-1])

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

    # static methods to get the closest object, approaching, object and running object
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

    # Use static methods of activity to get activity indicators
    def get_activity_indicators(self, doe_id, interval, verbose=True):
        other_tracks = list(range(self.n_obj))
        other_tracks.pop(doe_id)

        curr_position_list = self.timelines[doe_id].position
        other_initial_points = [self.timelines[idy].position[interval[0]] for idy in other_tracks]
        other_minus_ten_points = [self.timelines[idy].position[interval[0] - 10] for idy in other_tracks]
        other_plus_ten_points = [self.timelines[idy].position[interval[0] + 10] for idy in other_tracks]

        min_idx, min_dist = self.get_closest_point(curr_position_list[interval[0]], other_initial_points)
        pre_idx, max_pre_dist = self.get_approaching_object(curr_position_list[interval[0]], other_minus_ten_points,
                                                            other_initial_points)
        post_idx, max_post_dist = self.get_approaching_object(curr_position_list[interval[0]], other_initial_points,
                                                              other_plus_ten_points)
        run_idx, max_run_dist = self.get_running_object(curr_position_list[interval[0]], other_initial_points,
                                                        other_plus_ten_points)
        acceleration_c_list = list(map(abs, self.timelines[doe_id].acceleration[interval[0]+1:interval[1]]))
        acceleration_c = len([x for x in acceleration_c_list if x <= 1]) / (interval[1] - interval[0])

        if verbose:
            print('--------------', '\n',
                  'Doe: ', doe_id, '\n',
                  'Frames:', interval, '\n',
                  'Closest Animal: ', other_tracks[min_idx], min_dist, '\n',
                  'Pre-Approaching Animal: ', other_tracks[pre_idx], max_pre_dist, '\n',
                  'Post-Approaching Animal: ', other_tracks[post_idx], max_post_dist, '\n',
                  'Running Animal: ', other_tracks[run_idx], max_run_dist, '\n',
                  'Acceleration', acceleration_c, '\n',
                  '--------------', '\n')

        return (other_tracks[min_idx], min_dist), (other_tracks[pre_idx], max_pre_dist), \
               (other_tracks[post_idx], max_post_dist), (other_tracks[run_idx], max_run_dist), acceleration_c

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

    # Classify events within the set using activity indicators
    @staticmethod
    def check_cycle(graph):
        def has_cycles(n, s):
            if n in s:
                yield False, n
            else:
                yield True, n
                yield from [i for a, b in graph for i in has_cycles(b, s + [n]) if a == n]

        return not all(a for a, _ in has_cycles(graph[0][0], []))

    @staticmethod
    def check_run_off(p, a, b, r):
        if a[1] >= 20:
            return a[0]
        elif b[1] >= 20:
            return b[0]
        elif r[1] < 10 and a[1] + b[1] > 20:
            if a[1] > b[1]:
                return a[0]
            else:
                return b[0]
        elif r[1] < 10 and p[1] < 50:
            return p[0]
        else:
            return -1

    @staticmethod
    def check_perpetrate(r):
        if r[1] >= 10:
            return r[0]
        else:
            return -1

    def classify_events(self, verbose=True):
        event_dict = self.aggregate_active_moments(factor=10)
        df = pd.DataFrame({'Perpetrator': [], 'Target': [], 'Interval Start': [], 'Interval End': []})

        for key in event_dict:
            temp_event_tuples = []
            temp_event_intervals = []
            for activity in sorted(event_dict[key], key=lambda x: (x[0])):
                set_participants = set([i[2] for i in event_dict[key]])
                c, pre, post, run, a_c = self.get_activity_indicators(activity[2], activity[0:2])
                run_off_id = self.check_run_off(c, pre, post, run)
                chaser_id = self.check_perpetrate(run)

                if chaser_id in set_participants and chaser_id != -1:
                    temp_tuple = (activity[2], chaser_id)
                    if temp_tuple not in temp_event_tuples:
                        temp_event_tuples.append(temp_tuple)
                        temp_event_intervals.append(activity[0:2])
                        if self.check_cycle(temp_event_tuples):
                            temp_event_tuples.pop(-1)
                            temp_event_intervals.pop(-1)

                if (run_off_id in set_participants or c[0] == run_off_id) and run_off_id != -1 and a_c < 0.5:
                    temp_tuple = (run_off_id, activity[2])
                    if temp_tuple not in temp_event_tuples:
                        temp_event_tuples.append(temp_tuple)
                        temp_event_intervals.append(activity[0:2])
                        if self.check_cycle(temp_event_tuples):
                            temp_event_tuples.pop(-1)
                            temp_event_intervals.pop(-1)

            for idx in range(len(temp_event_tuples)):
                df.loc[df.shape[0]] = [temp_event_tuples[idx][0], temp_event_tuples[idx][1],
                                       temp_event_intervals[idx][0], temp_event_intervals[idx][1]]

            if verbose:
                print(temp_event_intervals)
                print(temp_event_tuples)
                print(temp_event_intervals)
                print(df)

        return df.sort_values(by=['Interval Start'], ignore_index=True)

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
