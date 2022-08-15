def get_activity_indicators_dev(self):
    for idx in range(self.n_obj):
        other_tracks = list(range(self.n_obj))
        other_tracks.pop(idx)
        curr_position_list = self.timelines[idx].position
        for active_moment in self.timelines[idx].active_moments:
            other_initial_points = [self.timelines[idy].position[active_moment[0]] for idy in other_tracks]
            other_minus_ten_points = [self.timelines[idy].position[active_moment[0] - 10] for idy in other_tracks]
            other_plus_ten_points = [self.timelines[idy].position[active_moment[0] + 10] for idy in other_tracks]

            min_idx, min_dist = self.get_closest_point(curr_position_list[active_moment[0]], other_initial_points)

            pre_idx, max_pre_dist = self.get_approaching_object(curr_position_list[active_moment[0]],
                                                                other_minus_ten_points,
                                                                other_initial_points)
            post_idx, max_post_dist = self.get_approaching_object(curr_position_list[active_moment[0]],
                                                                  other_initial_points,
                                                                  other_plus_ten_points)
            run_idx, max_run_dist = self.get_running_object(curr_position_list[active_moment[0]],
                                                            other_initial_points,
                                                            other_plus_ten_points)

            print('--------------', '\n',
                  'Doe: ', idx, '\n',
                  'Frames:', active_moment, '\n',
                  'Closest Animal: ', other_tracks[min_idx], min_dist, '\n',
                  'Pre-Approaching Animal: ', other_tracks[pre_idx], max_pre_dist, '\n',
                  'Post-Approaching Animal: ', other_tracks[post_idx], max_post_dist, '\n',
                  'Running Animal: ', other_tracks[run_idx], max_run_dist, '\n',
                  '--------------', '\n')

def get_approximate_trajectory(self):
    z = np.polyfit(x, y, 3)
    x = np.arange(min(x), max(x))
    y = np.polyval(z, x)
    y = 480 - y
    plt.plot(x, y)
    plt.xlim(0, 640)
    plt.ylim(0, 480)
    plt.show()


temp_tuple = (activity[2], chaser_id)
other_acc = self.check_perpetrator_acceleration(temp_tuple[1], activity[0:2])
track_overlap = self.check_track_overlap(temp_tuple[0], temp_tuple[1], activity[0:2])
displacement = self.get_displacement(activity[2], activity[0:2])

if temp_tuple not in temp_event_tuples and other_acc < 0.8 and track_overlap > 50 \
        and displacement > 10:
    temp_event_tuples.append(temp_tuple)
    temp_event_intervals.append(activity[0:2])
    if self.check_cycle(temp_event_tuples):
        # dyadic cycle check
        if (temp_tuple[1], temp_tuple[0]) in temp_event_tuples:
            idx = temp_event_tuples.index((temp_tuple[1], temp_tuple[0]))
            print(idx, run[1], temp_run[idx])
            if run[1] > temp_run[idx]:
                temp_event_tuples.pop(idx)
            else:
                temp_event_tuples.pop(-1)
                temp_event_intervals.pop(-1)
        else:
            temp_event_tuples.pop(-1)
            temp_event_intervals.pop(-1)

if temp_tuple in temp_event_tuples:
    temp_event_intervals[temp_event_tuples.index(temp_tuple)][1] = activity[1]

temp_tuple = (run_off_id, activity[2])
other_acc = self.check_perpetrator_acceleration(temp_tuple[0], activity[0:2])
track_overlap = self.check_track_overlap(temp_tuple[0], temp_tuple[1], activity[0:2])

# displacement = self.get_displacement(run_off_id, activity[0:2])
# print('aha', displacement)

displacement = self.get_displacement(activity[2], activity[0:2])

if temp_tuple not in temp_event_tuples and other_acc < 0.8 \
        and track_overlap > 50 and displacement > 20:
    temp_event_tuples.append(temp_tuple)
    temp_event_intervals.append(activity[0:2])
    if self.check_cycle(temp_event_tuples):
        temp_event_tuples.pop(-1)
        temp_event_intervals.pop(-1)

if temp_tuple in temp_event_tuples:
    temp_event_intervals[temp_event_tuples.index(temp_tuple)][1] = activity[1]

# return (other_tracks[min_idx], min_dist), (other_tracks[pre_idx], max_pre_dist), \
#       (other_tracks[post_idx], max_post_dist), (other_tracks[run_idx], max_run_dist), acceleration_c, \
#       third_dist, (other_tracks[post_run_idx], post_max_run_dist)