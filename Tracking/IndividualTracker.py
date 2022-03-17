import json
from scipy.spatial import distance
from IndividualTrack import Track
from DistanceTracker import DistanceTracker


class RabbitTracker:

    # init function
    def __init__(self, dict_file, n_obj=4):

        self.dict_file = dict_file
        with open(self.dict_file, 'r') as f:
            self.dict = json.load(f)

        self.n_obj = n_obj
        self.tracks = []
        for track_id in range(n_obj):
            self.tracks.append(Track(track_id))

    # get functions

    def get_bbox(self, frame):
        return self.dict[str(frame)]['bbox']

    def get_keypoints(self, frame):
        return self.dict[str(frame)]['keypoints']

    def get_confidence(self, frame):
        return self.dict[str(frame)]['confidence']

    def get_classification(self, frame):
        return self.dict[str(frame)]['classification']

    @staticmethod
    def get_centroid_bbox(bbox):
        return int(bbox[0] + bbox[2] / 2), int(bbox[1] + bbox[3] / 2)

    def get_track(self, track_id):
        return self.tracks[track_id]

    def get_track_doe_id(self, track_id):
        return self.tracks[track_id].get_track_id()

    # calculate Euclidian distance between points

    @staticmethod
    def calculate_euclidian_distance(last_point, new_points):
        distance_list = []
        for centroid in new_points:
            distance_list.append(distance.euclidean(last_point, centroid))
        return distance_list

    @staticmethod
    def get_minimum_distance(distance_list):
        minimum_distance = None
        minimum_distance_centroid = None
        for idx, dist in enumerate(distance_list):
            if minimum_distance is None or dist < minimum_distance:
                minimum_distance = dist
                minimum_distance_centroid = idx
        return minimum_distance_centroid, minimum_distance

    def get_closest_centroid(self, prev_point, frame):
        minimum_distance = None
        minimum_idx = None
        for idx, bbox in enumerate(self.get_bbox(frame)):
            temp_dist = distance.chebyshev(prev_point, self.get_centroid_bbox(bbox))
            if minimum_idx is None or temp_dist < minimum_distance:
                minimum_distance = temp_dist
                minimum_idx = idx
        return self.get_centroid_bbox(self.get_bbox(frame)[minimum_idx]), minimum_distance

    @staticmethod
    def get_extrapolation_between_points(prev_point, next_point, n_frames):
        x_increment = (next_point[0] - prev_point[0]) / (n_frames + 1)
        y_increment = (next_point[1] - prev_point[1]) / (n_frames + 1)
        extrapolate_points_arr = []
        for frame in range(n_frames):
            extrapolate_points_arr.append((prev_point[0] + (x_increment * (frame + 1)),
                                           prev_point[1] + (y_increment * (frame + 1))))
        return extrapolate_points_arr

    # generate tracks

    def initialize_track(self, doe_id, threshold):
        self.get_track(doe_id).initialize_point_dict(len(self.dict))

        for c in range(len(self.dict)):
            top_prob = 0.0
            for idx, element in enumerate(self.get_classification(c)):
                if element[0] == doe_id and element[1] > threshold and element[1] > top_prob:
                    top_prob = element[1]
                    self.get_track(doe_id).add_point(c, self.get_centroid_bbox(self.get_bbox(c)[idx]))

                    if c == 0 or self.get_track(doe_id).get_point(c - 1)['confidence'] is None:
                        self.get_track(doe_id).add_confidence(c, 0.20)
                    elif self.get_track(doe_id).get_point(c - 1)['confidence'] < 1.0:
                        self.get_track(doe_id).add_confidence(c,
                                                              (self.get_track(doe_id).get_point(c - 1)['confidence'] +
                                                               0.20))
                    else:
                        self.get_track(doe_id).add_confidence(c, 1.00)

                    if self.get_track(doe_id).get_point(c)['confidence'] >= 0.8:
                        self.get_track(doe_id).get_point(c - 1)['confidence'] = 0.8
                        self.get_track(doe_id).get_point(c - 2)['confidence'] = 0.8
                        self.get_track(doe_id).get_point(c - 3)['confidence'] = 0.8

        for key in self.get_track(doe_id).point_dict:
            if self.get_track(doe_id).point_dict[key]['confidence'] is not None and \
                    self.get_track(doe_id).point_dict[key]['confidence'] < 0.80:
                self.get_track(doe_id).point_dict[key] = {
                    'point': None,
                    'confidence': None
                }

    # fill initial frames
    def get_initial_points(self):
        initial_points = {}
        for track_id in range(self.n_obj):
            track_point_dict = self.get_track(track_id).point_dict
            for key in track_point_dict:
                if track_point_dict[key]['point'] is not None:
                    initial_points[track_id] = int(key)
                    break
        return initial_points

    def fill_initial_points(self):
        initial_points_dict = self.get_initial_points()
        while len(initial_points_dict) > 0:
            min_initial_track = None
            for key in initial_points_dict:
                if min_initial_track is None or initial_points_dict[min_initial_track] > initial_points_dict[key]:
                    min_initial_track = key
            if initial_points_dict[min_initial_track] > 0:
                point = self.get_track(min_initial_track).point_dict[
                    str(initial_points_dict[min_initial_track])]['point']
                confidence_coefficient = 1 / int(initial_points_dict[min_initial_track])
                for idx in range(int(initial_points_dict[min_initial_track])):
                    self.get_track(min_initial_track).add_point(idx, point)
                    self.get_track(min_initial_track).add_confidence(idx, round((idx + 1) * confidence_coefficient, 4))
            del initial_points_dict[min_initial_track]
        return initial_points_dict

    # fill final frames
    def get_final_points(self):
        final_points = {}
        for track_id in range(self.n_obj):
            track_point_dict = self.get_track(track_id).point_dict
            for key in reversed(list(track_point_dict.keys())):
                if track_point_dict[key]['point'] is not None:
                    final_points[track_id] = int(key)
                    break
        return final_points

    def fill_final_points(self):
        final_points_dict = self.get_final_points()
        while len(final_points_dict) > 0:
            max_final_track = None
            for key in final_points_dict:
                if max_final_track is None or final_points_dict[max_final_track] < final_points_dict[key]:
                    max_final_track = key
            if final_points_dict[max_final_track] < len(self.dict):
                point = self.get_track(max_final_track).point_dict[
                    str(final_points_dict[max_final_track])]['point']
                confidence_coefficient = 1 / (len(self.dict) - int(final_points_dict[max_final_track]))
                for idx in range(int(final_points_dict[max_final_track]), len(self.dict)):
                    self.get_track(max_final_track).add_point(idx, point)
                    self.get_track(max_final_track).add_confidence(idx,
                                                                   round((1 - ((idx -
                                                                                int(final_points_dict[
                                                                                        max_final_track])) *
                                                                               confidence_coefficient)), 4))
            del final_points_dict[max_final_track]
        return final_points_dict

    # fill in-between classified
    def fill_in_between_classified_points(self, doe_id):
        track_point_dict = self.get_track(doe_id).point_dict
        prev_valid_frame = None
        last_valid_frame = None
        lost_count = 0

        for key in reversed(list(track_point_dict.keys())):
            if track_point_dict[key]['point'] is not None:
                last_valid_frame = int(key)
                break

        for key in track_point_dict:
            if track_point_dict[key]['point'] is not None:
                if prev_valid_frame is not None and lost_count > 0:
                    candidate_track = self.find_candidate_track(int(prev_valid_frame),
                                                                int(key),
                                                                track_point_dict[prev_valid_frame]['point'],
                                                                track_point_dict[key]['point'],
                                                                doe_id)
                    if candidate_track is None:
                        ext_arr = self.get_extrapolation_between_points(track_point_dict[prev_valid_frame]['point'],
                                                                        track_point_dict[key]['point'],
                                                                        lost_count)
                        confidence_avg = (track_point_dict[prev_valid_frame]['confidence'] +
                                          track_point_dict[key]['confidence']) / 2
                        for idx, ext_point in enumerate(ext_arr):
                            temp_centroid, temp_dist = self.get_closest_centroid(ext_point,
                                                                                 int(prev_valid_frame) + idx + 1)
                            if temp_dist < 5:
                                self.get_track(doe_id).add_point(int(prev_valid_frame) + idx + 1, temp_centroid)
                                self.get_track(doe_id).add_confidence(int(prev_valid_frame) + idx + 1,
                                                                      confidence_avg - temp_dist * 0.05)
                            else:
                                self.get_track(doe_id).add_point(int(prev_valid_frame) + idx + 1, ext_point)
                                self.get_track(doe_id).add_confidence(int(prev_valid_frame) + idx + 1,
                                                                      confidence_avg - (idx / lost_count) * 0.25)
                    else:
                        for idx, (centroid, confidence) in enumerate(candidate_track):
                            self.get_track(doe_id).add_point(int(prev_valid_frame) + idx, centroid)
                            self.get_track(doe_id).add_confidence(int(prev_valid_frame) + idx, confidence)
                prev_valid_frame = key
                lost_count = 0
            else:
                if prev_valid_frame is None or int(key) > last_valid_frame:
                    continue
                else:
                    lost_count += 1

    # Distance based candidate tracks
    def find_candidate_track(self, initial_frame, final_frame, initial_point, final_point, doe_id):
        partial_dict = {}
        for idx in range(initial_frame, final_frame):
            partial_dict[str(idx - initial_frame)] = self.dict[str(idx)]

        distance_tracker = DistanceTracker(partial_dict, n_obj=self.n_obj)
        distance_tracker.generate_tracks()

        max_combined_prob = 0.0
        single_track_validation_sum = 0
        best_track_id = None
        for track_id in range(self.n_obj):
            initial_point_dist = distance.euclidean(initial_point, distance_tracker.get_track(track_id)[1][0])
            final_point_dist = distance.euclidean(final_point, distance_tracker.get_track(track_id)[-2][0])
            combined_dist = initial_point_dist + final_point_dist
            candidate_prob = self.validate_candidate_track(partial_dict,
                                                           distance_tracker.get_track(track_id)[1:],
                                                           doe_id)
            single_track_validation_sum += candidate_prob
            if best_track_id is None or candidate_prob > max_combined_prob:
                best_track_id = track_id
                max_combined_prob = candidate_prob
        single_track_validation_ratio = max_combined_prob/single_track_validation_sum

        all_track_validation_sum = 0.0
        for idx in range(self.n_obj):
            all_track_validation_sum += self.validate_candidate_track(partial_dict,
                                                                      distance_tracker.get_track(best_track_id)[1:],
                                                                      idx)
        validation_ratio = self.validate_candidate_track(partial_dict,
                                                         distance_tracker.get_track(best_track_id)[1:],
                                                         doe_id) / all_track_validation_sum

        if (validation_ratio + single_track_validation_ratio) > 1:
            forward_stable_track = self.stabilize_candidate_track(partial_dict,
                                                                  distance_tracker.get_track(best_track_id)[1:], doe_id)
            forward_stable_track.reverse()
            backward_stable_track = self.stabilize_candidate_track(partial_dict, forward_stable_track, doe_id)
            backward_stable_track.reverse()
            return backward_stable_track
        else:
            return None

    def validate_candidate_track(self, track_dict, track, doe_id):
        sum_class = 0.0
        count = 0
        for idx, (centroid, confidence) in enumerate(track):
            if confidence == 1:
                for idy, bbox in enumerate(track_dict[str(idx)]['bbox']):
                    if centroid == self.get_centroid_bbox(bbox):
                        count += 1
                        if track_dict[str(idx)]['classification'][idy][0] == doe_id:
                            sum_class += track_dict[str(idx)]['classification'][idy][1]
        if count == 0:
            return 0
        else:
            return sum_class / count

    def stabilize_candidate_track(self, track_dict, track, doe_id):
        final_track = track.copy()
        prev_valid_point = None
        last_point = len(track) - 1

        for idx, (centroid, confidence) in enumerate(track):
            if confidence == 1:
                for idy, bbox in enumerate(track_dict[str(idx)]['bbox']):
                    if centroid == self.get_centroid_bbox(bbox):
                        point_doe_id = track_dict[str(idx)]['classification'][idy][0]
                        point_classification_prob = track_dict[str(idx)]['classification'][idy][1]
                        if point_doe_id == doe_id and point_classification_prob > 0.95:
                            if prev_valid_point is None:
                                prev_valid_point = idx
                            else:
                                if idx - prev_valid_point > 1:
                                    ext_trac = self.get_extrapolation_between_points(track[prev_valid_point][0],
                                                                                     track[idx][0],
                                                                                     idx - prev_valid_point - 1)
                                    for idz, point in enumerate(range(prev_valid_point+1, idx)):
                                        temp_dist = distance.euclidean(ext_trac[idz], track[point][0])
                                        if temp_dist < 20.0:
                                            final_track[point] = track[point][0], round(0.95 - temp_dist/100, 2)
                                        else:
                                            ext_trac[idz] = round(ext_trac[idz][0]), round(ext_trac[idz][1])
                                            final_track[point] = ext_trac[idz], round(0.95 -
                                                                                      (idz+1)/len(ext_trac) * 0.1, 2)
                                prev_valid_point = idx

        if prev_valid_point is not None and last_point - prev_valid_point > 1:
            ext_trac = self.get_extrapolation_between_points(track[prev_valid_point][0],
                                                             track[last_point][0],
                                                             last_point - prev_valid_point - 1)
            for idz, point in enumerate(range(prev_valid_point + 1, last_point)):
                temp_dist = distance.euclidean(ext_trac[idz], track[point][0])
                if temp_dist < 20.0:
                    final_track[point] = (track[point][0], round(0.95 - temp_dist / 100, 2))
                else:
                    ext_trac[idz] = round(ext_trac[idz][0]), round(ext_trac[idz][1])
                    final_track[point] = ext_trac[idz], round(0.95 - (idz + 1) / len(ext_trac) * 0.1, 2)
        return final_track



    def find_candidate_tracks2(self, doe_id):
        track_point_dict = self.get_track(doe_id).point_dict
        prev_valid_frame = None

        for key in track_point_dict:
            if track_point_dict[key]['point'] is not None:
                if prev_valid_frame is not None:
                    last_centroid = track_point_dict[prev_valid_frame]['point']
                    current_centroid = track_point_dict[key]['point']

                prev_valid_frame = key

    def add_coord_track(self, frame):
        distance_dict = self.calculate_euclidian_distance(self.tracks,
                                                          self.get_centroid_bbox(self.get_bbox(frame)))
        updated_tracks = []
        while len(distance_dict) > 0:
            minimum_distance_track, minimum_distance_centroid, min_distance = self.get_minimum_distance(distance_dict)
            if minimum_distance_track is not None:
                self.tracks[minimum_distance_track].add_point((self.get_centroid_bbox(
                    self.get_bbox(frame))[minimum_distance_centroid], 1))
                self.delete_track_centroid(distance_dict, minimum_distance_track, minimum_distance_centroid)
                updated_tracks.append(minimum_distance_track)
            else:
                break

        for track in self.tracks:
            if track.track_id not in updated_tracks:
                track.add_point((track.get_last_point()[0], 0))

    def generate_tracks(self):
        self.initialize_tracks()
        for frame in self.dict:
            if frame != 0:
                self.add_coord_track(frame)
