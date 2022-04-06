# Import libraries / classes
import json
import numpy as np
from scipy.spatial import distance
from IndividualTrack import Track
from DistanceTracker import DistanceTracker


# Define RabbitTracker class
class RabbitTracker:
    # init function
    """
    Takes json file in the format of
    {str(frame_no): {
                        bbox: [...],
                        keypoints: [...],
                        confidence; [...],
                        classification: [...],
                    },...}
    Definition of number of objects is necessary and tracker takes power from the prior knowledge of tracks.
    Refer to the Track class for Track objects.
    """

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

    # Calculate Euclidian distance between points
    """
    Computes:
        Euclidian distance from single point to multiple points using SciPy implementation of Euclidean distance
    Params:
        last_point: point in format of tuple (x,y) from a track 
        new_points: list of tuples in format of [(x1,y1),...(xn,yn)] from new frame
    Returns:
        List (float) - distance of last_point with respect to the new_points  
    """

    @staticmethod
    def calculate_euclidian_distance(last_point, new_points):
        distance_list = []
        for centroid in new_points:
            distance_list.append(distance.euclidean(last_point, centroid))
        return distance_list

    # Get minimum distance point object from the distance list
    """
    Computes: 
        Object with the minimum distance return from the calculate_euclidian_distance()
    Params:
        distance_list: List (Float) - distance of last_point with respect to the new_points 
    Returns:
        1. point in format of tuple (x, y) which is closest to the previous given point 
        2. Euclidean distance (float)
    """

    @staticmethod
    def get_minimum_distance(distance_list):
        minimum_distance = None
        minimum_distance_centroid = None
        for idx, dist in enumerate(distance_list):
            if minimum_distance is None or dist < minimum_distance:
                minimum_distance = dist
                minimum_distance_centroid = idx
        return minimum_distance_centroid, minimum_distance

    # Get the closest centroid; a hybrid implementation of static methods with access to class functions
    """
    Computes: 
        Object with the minimum distance return from a given prev_point
    Params:
        prev_point: point in format of tuple (x,y) from a track 
        frame: frame number to retrieve objects (integer)
    Returns:
        1. point in format of tuple (x, y) which is closest to the previous given prev_point 
        2. Euclidean distance (float)
    """

    def get_closest_centroid(self, prev_point, frame):
        minimum_distance = None
        minimum_idx = None
        for idx, bbox in enumerate(self.get_bbox(frame)):
            temp_dist = distance.chebyshev(prev_point, self.get_centroid_bbox(bbox))
            if minimum_idx is None or temp_dist < minimum_distance:
                minimum_distance = temp_dist
                minimum_idx = idx
        return self.get_centroid_bbox(self.get_bbox(frame)[minimum_idx]), minimum_distance

    # Compute linear extrapolation between two points
    """
    Computes: 
        Array of points between two given points where the points are interpolated with a linear function
    Params:
        prev_point: point in format of tuple (x,y) from track idx 
        next_point: point in format of tuple (x,y) from track idx 
        n_frames: number of frames in format of integer
    Returns:
        List (Tuple) - the point tuples between prev_point and next_point with len=n_frames
    """

    @staticmethod
    def get_extrapolation_between_points(prev_point, next_point, n_frames):
        x_increment = (next_point[0] - prev_point[0]) / (n_frames + 1)
        y_increment = (next_point[1] - prev_point[1]) / (n_frames + 1)
        extrapolate_points_arr = []
        for frame in range(n_frames):
            extrapolate_points_arr.append((prev_point[0] + (x_increment * (frame + 1)),
                                           prev_point[1] + (y_increment * (frame + 1))))
        return extrapolate_points_arr

    # Initialization of Track objects
    """
    Computes: 
        Initialize the tracks by first filling with dictionaries in format of {'Point': None, 'Confidence': None}
        Search through all frames and assign the animal with highest classification (if larger than given threshold)
        to the track. Use confidence parameter to build certainty over consecutive frames. The assignment requires
        identification of animal with greater than equal to given threshold over 5 frames.  
    Params:
        doe_id: animal id number (uint8)
        threshold: prediction threshold for filtering uncertain classifications (float) with domain = [0, 1.0] 
    Returns:
        None
    """

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

    # Clean spontaneous occurrences after the track initialization
    """
    Computes: 
        Clear the spontaneous occurrences where the number of frames given as parameter is less than oe equal to the
        number of valid frames assigned during the initialization. Valid frame for a track is defined as 'Point' 
        is not None
    Params:
        n_frames: number of frames in format of integer
    Returns:
        None
    """

    def clean_in_between_occurrences(self, n_frames):
        for track_id in range(self.n_obj):
            track_point_dict = self.get_track(track_id).point_dict
            prev_key = None
            frame_count = 0
            for key in track_point_dict:
                if track_point_dict[key]['point'] is not None:
                    if frame_count == 0:
                        prev_key = key
                    frame_count += 1
                else:
                    if 0 < frame_count <= n_frames:
                        for idx in range(int(prev_key), int(key)):
                            self.get_track(track_id).point_dict[str(idx)] = {
                                'point': None,
                                'confidence': None
                            }
                    frame_count = 0

    # Helper function to get golden points
    """
    Computes: 
        Golden points are the frames where all objects are identified and tracks are updated accordingly 
    Params:
        None
    Returns:
        None
        * Prints the golden points for debugging
    """

    def get_gold_key_points(self):
        for point in self.dict:
            point_bool = True
            for track in range(self.n_obj):
                if self.get_track(track).get_point(point)['point'] is None:
                    point_bool = False
            if point_bool:
                print('Golden Point:', point)

    # Manipulating tracks using silver points
    """
    Computes: 
        Silver points are the frames where n-1 objects are identified accurately and tracks are updated accordingly.
        The object found but not identified is assigned the id number of the lost object where other n-1 objects
        are identified almost certainly with default classification probability > 0.99 
    Params:
        None
    Returns:
        None
        * Manipulates track objects directly
    """

    def fix_silver_key_points(self):
        for point in self.dict:
            silver_count = 0
            unfounded_track = -1
            for track in range(self.n_obj):
                if self.get_track(track).get_point(point)['point'] is None:
                    silver_count += 1
                    unfounded_track = track

            if silver_count == 1 and len(self.dict[point]['classification']) == self.n_obj:
                sum_99 = 0
                prob_arr = np.array([])
                for prob in self.dict[point]['classification']:
                    prob_arr = np.append(prob_arr, prob[1])
                    sum_99 = (prob_arr > 0.99).sum()
                if sum_99 == 3 and self.dict[point]['confidence'][prob_arr.argmin()] > 0.99:
                    self.get_track(unfounded_track).add_point(
                        point, self.get_centroid_bbox(self.dict[point]['bbox'][prob_arr.argmin()]))
                    self.get_track(unfounded_track).add_confidence(point, 0.8)

    # Get initial certain point
    """
    Computes: 
        The initial frame number where the object is identified accurately and injected into the Track object
    Params:
        None
    Returns:
        List (uint) - frame id of first certain points per track, len = number of tracks
    """

    # Fill initial frames
    def get_initial_points(self):
        initial_points = {}
        for track_id in range(self.n_obj):
            track_point_dict = self.get_track(track_id).point_dict
            for key in track_point_dict:
                if track_point_dict[key]['point'] is not None:
                    initial_points[track_id] = int(key)
                    break
        return initial_points

    # Get the closest initial Track object
    """
    Computes: 
        The initial frame number where the object is identified accurately and injected into the Track object
    Params:
        centroid_dict: dictionary that contains initial identification points in tuple (x,y)
        doe_id: animal identification number (uint8) with domain = [0, inf)
        unfounded_count = integer (uint8) representing count objects that have been not initialized at frame 0
    Returns:
        1. List of tuples (points in format of (x,y)) that is the best track associated with given animal id
        2. Dictionary of updated centroids
    """

    def get_closest_initial_track(self, centroid_dict, doe_id, unfounded_count):
        distance_tracker_obj = DistanceTracker({}, n_obj=self.n_obj)
        point_dict = self.get_initial_points()
        distance_dict = {}
        partial_dict = {}
        for idx in range(0, point_dict[doe_id]):
            partial_dict[str(idx)] = self.dict[str(idx)]
        distance_tracker = DistanceTracker(partial_dict, n_obj=self.n_obj)
        distance_tracker.generate_tracks()
        for track_idy in range(self.n_obj):
            x_array = np.array([])
            y_array = np.array([])
            for coord in distance_tracker.get_track(track_idy)[1:]:
                x_array = np.append(x_array, coord[0][0])
                y_array = np.append(y_array, coord[0][1])
            for track_idz in centroid_dict:
                if track_idy not in distance_dict:
                    distance_dict[track_idy] = {}
                if doe_id != track_idz:
                    distance_dict[track_idy][track_idz] = distance.euclidean(
                        (np.median(x_array), np.median(y_array)), centroid_dict[track_idz])
        for idx in range(self.n_obj - unfounded_count):
            track_id, doe_idx, min_dist = distance_tracker_obj.get_minimum_distance(distance_dict)
            distance_tracker_obj.delete_track_centroid(distance_dict, track_id, doe_idx)

        if unfounded_count == 1:
            x_array = np.array([])
            y_array = np.array([])
            for coord in distance_tracker.get_track(list(distance_dict)[0])[1:]:
                x_array = np.append(x_array, coord[0][0])
                y_array = np.append(y_array, coord[0][1])
            centroid_dict[doe_id] = (np.median(x_array), np.median(y_array))
            return distance_tracker.get_track(list(distance_dict)[0])[1:], centroid_dict
        else:
            step_one_dict = self.dict[(str(int(point_dict[doe_id]) + 1))]
            min_idx = None
            max_score = None
            for idx, values in enumerate(step_one_dict['classification']):
                if values[0] == doe_id and (min_idx is None or max_score < values[1]):
                    min_idx = idx
                    max_score = values[1]
            bbox = step_one_dict['bbox'][min_idx]
            centroid_step_one = (int(bbox[0] + bbox[2] / 2), int(bbox[1] + bbox[3] / 2))

            best_track = None
            min_track_dist = None
            for candidate_track in distance_dict:
                x_array = np.array([])
                y_array = np.array([])
                for coord in distance_tracker.get_track(candidate_track)[1:]:
                    x_array = np.append(x_array, coord[0][0])
                    y_array = np.append(y_array, coord[0][1])
                temp_euclidean = distance.euclidean(centroid_step_one, (np.median(x_array), np.median(y_array)))
                if best_track is None or min_track_dist > temp_euclidean:
                    best_track = candidate_track
                    min_track_dist = temp_euclidean
            x_array = np.array([])
            y_array = np.array([])
            for coord in distance_tracker.get_track(best_track)[1:]:
                x_array = np.append(x_array, coord[0][0])
                y_array = np.append(y_array, coord[0][1])
            centroid_dict[doe_id] = (np.median(x_array), np.median(y_array))
            return distance_tracker.get_track(best_track)[1:], centroid_dict

    # Fill initial points using get_initial_points() and get_closest_initial_track()
    """
    Computes: 
       The initial centroid dictionary in order to pass into get_closest_initial_track() function. Depending on count
       of unfounded objects (1 or >1), call get_closest_initial_track() function. It fills in a descending order of
       frame id. In order words, initially fill the object that identified latest then second latest and so on. It is
       more robust to wrong identification where the later identified object inherit have more noisy probabilities. 
    Params:
        None
    Returns:
        None
        * Manipulates track objects directly
    """

    def _fill_initial_points(self):
        initial_points_dict = self.get_initial_points()
        initial_centroid_dict = {}
        initial_unfounded = 0

        for doe_id in initial_points_dict:
            initial_centroid_dict[doe_id] = None
            partial_dict = {}
            for idx in range(0, initial_points_dict[doe_id]):
                partial_dict[str(idx)] = self.dict[str(idx)]
            if len(partial_dict) == 0:
                initial_centroid_dict[doe_id] = self.get_track(doe_id).point_dict[str(0)]['point']

        for doe_id in initial_centroid_dict:
            if initial_centroid_dict[doe_id] is None:
                initial_unfounded += 1

        if initial_unfounded == 1:
            for doe_id in initial_centroid_dict:
                if initial_centroid_dict[doe_id] is None:
                    track, centroid_dict = self.get_closest_initial_track(initial_centroid_dict,
                                                                          doe_id,
                                                                          initial_unfounded)
                    for idy in range(0, initial_points_dict[doe_id]):
                        self.get_track(doe_id).add_point(idy, track[idy][0])
                        self.get_track(doe_id).add_confidence(idy, track[idy][1])

        if initial_unfounded > 1:
            unfounded = initial_unfounded
            while unfounded > 0:
                prior_doe_id = None
                max_point = None
                for doe_id in initial_points_dict:
                    if initial_points_dict[doe_id] != 0 and \
                            (max_point is None or max_point > initial_points_dict[doe_id]):
                        prior_doe_id = doe_id
                        max_point = initial_points_dict[doe_id]
                initial_points_dict[prior_doe_id] = 0
                initial_centroid_dict_temp = initial_centroid_dict.copy()
                for key in initial_centroid_dict:
                    if initial_centroid_dict[key] is None and key != prior_doe_id:
                        del initial_centroid_dict_temp[key]

                track, centroid_dict = self.get_closest_initial_track(initial_centroid_dict_temp,
                                                                      prior_doe_id,
                                                                      unfounded)
                for idy in range(0, max_point):
                    self.get_track(prior_doe_id).add_point(idy, track[idy][0])
                    self.get_track(prior_doe_id).add_confidence(idy, track[idy][1])
                initial_centroid_dict[prior_doe_id] = centroid_dict[prior_doe_id]
                unfounded -= 1

    # Get final certain point
    """
    Computes: 
        The final frame number where the object is identified accurately and injected into the Track object
    Params:
        None
    Returns:
        List (uint) - frame id of final certain points per track, len = number of tracks
    """

    def get_final_points(self):
        final_points = {}
        for track_id in range(self.n_obj):
            track_point_dict = self.get_track(track_id).point_dict
            for key in reversed(list(track_point_dict.keys())):
                if track_point_dict[key]['point'] is not None:
                    final_points[track_id] = int(key)
                    break
        return final_points

    # Fill final points using get_final_points()
    """
    Computes: 
       The final points that are left unidentified in the tracks are filled by repetition of the last valid point.
       Inherently, it assumes that the objects do not change their positions after last identification. The assumption
       works well in practice with minor dispositions.
    Params:
        None
    Returns:
        dictionary of final points that represents last valid of identification of each object in format of 
            {'Track ID': tuple point (x,y),...}
        * Manipulates track objects directly
    """

    def _fill_final_points(self):
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

    # find candidate track based on Distance tracker
    """
    Computes: 
       The fusion of classification (re-identification) of objects and distance of tracks with respective to objects
       happens in the computation. Based on median coordinates of a tracks generated between custom frames, the
       animal is validated with found track distance and lost number of frames. If a validated match found among
       the candidate tracks generated with Distance Tracker, function returns to the list of points else returns None.  
    Params:
        initial_frame: initial frame number to start distance based tracking algorithm (integer)
        final_frame: final frame number to stop distance based tracking algorithm (integer) > initial_frame
        doe_id: animal identification number (uint8) with domain = [0, inf)
    Returns:
        List of tuples (points in format of (x,y)) that is the best candidate track associated with given animal id
            OR None (if no candidate track can be validated) 
    """
    def _find_candidate_track(self, initial_frame, final_frame, doe_id):
        partial_dict = {}
        for idx in range(initial_frame, final_frame + 1):
            partial_dict[str(idx - initial_frame)] = self.dict[str(idx)]

        distance_tracker = DistanceTracker(partial_dict, n_obj=self.n_obj)
        distance_tracker.generate_tracks()

        track_dict = {}
        for track_id in list(range(self.n_obj)):
            x_array = np.array([])
            y_array = np.array([])
            total_confidence = np.array([])
            for coord in distance_tracker.get_track(track_id)[1:]:
                x_array = np.append(x_array, coord[0][0])
                y_array = np.append(y_array, coord[0][1])
                total_confidence = np.append(total_confidence, coord[1])
            track_dict[track_id] = {
                'x_median': np.median(x_array),
                'y_median': np.median(y_array),
                'confidence': np.sum(total_confidence) / len(total_confidence)
            }

        does = list(range(self.n_obj))
        for doe_idx in list(range(self.n_obj)):
            if doe_idx != doe_id and self.get_track(doe_idx).get_point(initial_frame)['point'] is not None and \
                    self.get_track(doe_idx).get_point(final_frame)['point'] is not None:
                avg_point = ((self.get_track(doe_idx).get_point(initial_frame)['point'][0] +
                              self.get_track(doe_idx).get_point(final_frame)['point'][0]) / 2,
                             (self.get_track(doe_idx).get_point(initial_frame)['point'][1] +
                              self.get_track(doe_idx).get_point(final_frame)['point'][1]) / 2)

                for track_id in list(range(self.n_obj)):
                    if track_id in track_dict.keys() and avg_point is not None:
                        temp_dist = distance.euclidean(avg_point,
                                                       (track_dict[track_id]['x_median'],
                                                        track_dict[track_id]['y_median']))
                        if track_dict[track_id]['confidence'] > 0.9 and temp_dist < 20:
                            does.remove(int(doe_idx))
                            del track_dict[track_id]

        if len(does) == 1 and does[0] == doe_id:
            final_track_id = list(track_dict.items())[0][0]
            return distance_tracker.get_track(final_track_id)[1:]
        else:
            for track_idx in track_dict:
                initial_dist = distance.euclidean(distance_tracker.get_track(track_idx)[1:][0][0],
                                                  self.get_track(doe_id).get_point(initial_frame)['point'])
                final_dist = distance.euclidean(distance_tracker.get_track(track_idx)[1:][-1][0],
                                                self.get_track(doe_id).get_point(final_frame)['point'])
                if (final_frame - initial_frame) <= 50 and (initial_dist + final_dist) < 10:
                    count_overlap = 0
                    for idx, temp_point in enumerate(distance_tracker.get_track(track_idx)[1:]):
                        for doe_idy in range(self.n_obj):
                            if self.get_track(doe_idy).get_point(initial_frame + idx)['point'] is not None and \
                                    distance.euclidean(temp_point[0], self.get_track(
                                        doe_idy).get_point(initial_frame + idx)['point']) < 5:
                                count_overlap += 1
                    if count_overlap <= 2:
                        return distance_tracker.get_track(track_idx)[1:]
            return None

    # Fill the in between points where candidate track can be found
    """
    Computes: 
       Loop over every Track object and fill it with not None candidate track using _find_candidate_track() function.
    Params:
        None
    Returns:
        None
        * Manipulates track objects directly
    """
    def fill_in_between_unknown_single(self):
        for track_id in range(self.n_obj):
            track_point_dict = self.get_track(track_id).point_dict
            prev_key = None
            for key in track_point_dict:
                if track_point_dict[key]['point'] is not None:
                    if prev_key is None:
                        prev_key = key
                    else:
                        if int(key) - int(prev_key) > 1:
                            track = self._find_candidate_track(int(prev_key), int(key), track_id)
                            if track is not None:
                                for in_between_key in range(1, int(key) - int(prev_key)):
                                    self.get_track(track_id).add_point(int(prev_key) + in_between_key,
                                                                       track[in_between_key][0])
                                    self.get_track(track_id).add_confidence(int(prev_key) + in_between_key,
                                                                            track[in_between_key][1])
                        prev_key = key

    # Fill in-between classified anchor points with candidate track and linear interpolation
    """
    Computes: 
       For a given animal id within domain of [0,...,n_obj-1], fill points in between of anchor points identified
       using the classification model. The classification points act as re-identification moments and in-between points
       are uncertain. The function tries to find a candidate track using Distance-based tracker, if returns None, it
       interpolates linearly between two anchor points. Lastly, if the consecutive points are spatially close, 
       fill in-between with previous valid point and confidence. 
    Params:
        doe_id: animal id number (uint8)
    Returns:
        None
        * Manipulates track objects directly
    """
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
                    pp_distance = distance.euclidean(track_point_dict[prev_valid_frame]['point'],
                                                     track_point_dict[key]['point'])
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
                    elif pp_distance < 10:
                        for idx in range(int(prev_valid_frame), int(key)):
                            self.get_track(doe_id).add_point(idx, track_point_dict[prev_valid_frame]['point'])
                            self.get_track(doe_id).add_confidence(idx, track_point_dict[prev_valid_frame]['confidence'])
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

    # Find candidate track based on Distance tracker - initial and final point enhancement
    """
    Computes: 
       The fusion of classification (re-identification) of objects and distance of tracks with respective to objects
       happens in the computation. Based on coordinates of a tracks generated between custom frames, the
       animal is validated with classification probability ratios of both single track and multi-track.
       A simple sum of these two validation ratios needs to be larger than 1 to return a candidate track.
       If a validated match found among the candidate tracks generated with Distance Tracker, 
       function returns to the list of points else returns None.  
    Params:
        initial_frame: initial frame number to start distance based tracking algorithm (integer)
        final_frame: final frame number to stop distance based tracking algorithm (integer) > initial_frame
        initial_point: point in format of tuple (x,y) from a track 
        final_point: point in format of tuple (x,y) from a track 
        doe_id: animal identification number (uint8) with domain = [0, inf)
    Returns:
        List of tuples (points in format of (x,y)) that is the best candidate track associated with given animal id
            OR None (if no candidate track can be validated) 
    """
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
            combined_dist += combined_dist  # Obsolete
            candidate_prob = self.validate_candidate_track(partial_dict,
                                                           distance_tracker.get_track(track_id)[1:],
                                                           doe_id)
            single_track_validation_sum += candidate_prob
            if best_track_id is None or candidate_prob > max_combined_prob:
                best_track_id = track_id
                max_combined_prob = candidate_prob
        single_track_validation_ratio = 0
        if single_track_validation_sum != 0:
            single_track_validation_ratio = max_combined_prob / single_track_validation_sum

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

    # Validation of candidate track based on Distance tracker
    """
    Computes: 
       Computes the ratio of sum(classification scores | doe_id) and total number of the animal is identified
    Params:
        track_dict: sub-dictionary of original dictionary attribute for given frames in find_candidate_track()
        track: List of tuples (points in format of (x,y)) 
        doe_id: animal identification number (uint8) with domain = [0, inf)
    Returns:
        Normalised float number OR 0 (to avoid zeroDivide error) 
    """
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

    # Stabilization of candidate track based on Distance tracker
    """
    Computes: 
       Stabilize the given track with respect to the point and confidence values. 
    Params:
        track_dict: sub-dictionary of original dictionary attribute for given frames in find_candidate_track()
        track: List of tuples (points in format of (x,y)) 
        doe_id: animal identification number (uint8) with domain = [0, inf)
    Returns:
        List of tuples (points in format of (x,y)) 
    """
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
                                    for idz, point in enumerate(range(prev_valid_point + 1, idx)):
                                        temp_dist = distance.euclidean(ext_trac[idz], track[point][0])
                                        if temp_dist < 20.0:
                                            final_track[point] = track[point][0], round(0.95 - temp_dist / 100, 2)
                                        else:
                                            ext_trac[idz] = round(ext_trac[idz][0]), round(ext_trac[idz][1])
                                            final_track[point] = ext_trac[idz], round(0.95 -
                                                                                      (idz + 1) / len(ext_trac) * 0.1,
                                                                                      2)
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
        final_track = self.clean_candidate_track(final_track, 2, 2)
        return final_track

    # Alteration of invalid points from candidate track based on Distance tracker
    """
    Computes: 
       Cleans the candidate track from (-1, -1) points or points with 0 confidence attached. It assigns previous
       valid point and confidence in such occasions.
    Params:
        track: List of tuples (points in format of (x,y)) 
        min_prev: minimum previous count of invalid points
        min_lead: minimum next count of invalid points
    Returns:
        List of tuples (points in format of (x,y)) 
    """
    @staticmethod
    def clean_candidate_track(track, min_prev, min_lead):
        prev_broken_count = 0
        in_between_idx = []

        for idx, (point, confidence) in enumerate(track):
            if point == (-1, -1) or confidence == 0:
                prev_broken_count += 1
                in_between_idx.append(idx)
            else:
                if prev_broken_count >= min_prev:
                    for idy in range(min_lead):
                        if track[idx + idy + 1][0] == (-1, -1) or track[idx + idy + 1][1] == 0:
                            in_between_idx.append(idx)
                            break
                prev_broken_count = 0
        in_between_idx.reverse()

        for idx in in_between_idx:
            if idx != len(track) - 1:
                track[idx] = track[idx + 1]

        return track

    # Alteration of invalid points from candidate track based on Distance tracker
    """
    Computes: 
       Stabilize the tracks in batch operation using spatial proximity of consecutive point assignments.
    Params:
        window: integer value of window of frames to stabilize with domain = [0, len(self.dict))
        dist_threshold: float number that defines stabilization proximity
    Returns:
        None
        * Manipulates track objects directly
    """
    def stabilize_tracks(self, window, dist_threshold):
        for track_id in range(self.n_obj):
            track_point_dict = self.get_track(track_id).point_dict
            for key in track_point_dict:
                if int(key) < len(track_point_dict) - window - 1:
                    step_zero_none_check = track_point_dict[key]['point']
                    step_one_none_check = track_point_dict[str(int(key) + 1)]['point']
                    if step_zero_none_check is not None and step_one_none_check is not None:
                        initial_dist = distance.euclidean(track_point_dict[key]['point'],
                                                          track_point_dict[str(int(key) + 1)]['point'])
                        if initial_dist > dist_threshold:
                            for idy in range(window - 1):
                                plus_one_none_check = track_point_dict[str(idy + int(key) + 1)]['point']
                                plus_two_none_check = track_point_dict[str(idy + int(key) + 2)]['point']
                                if plus_one_none_check is not None and plus_two_none_check is not None:
                                    abs_dist_diff = abs(initial_dist -
                                                        distance.euclidean(
                                                            track_point_dict[str(idy + int(key) + 1)]['point'],
                                                            track_point_dict[str(idy + int(key) + 2)]['point']))
                                    if abs_dist_diff < 15:
                                        for idz in range(idy + 1):
                                            self.get_track(track_id).add_point(int(key) + idz + 1,
                                                                               track_point_dict[key]['point'])
                        if initial_dist < 5:
                            self.get_track(track_id).add_point(int(key) + 1, track_point_dict[key]['point'])

    # Avoid overlapping assignment (Perfect overlap)
    """
    Computes: 
       Clears the exactly overlapping tracks. Some of the fill in between tracks assign same animal for a short
       duration to the more than single track. To avoid this exact assignment, the function calculates the most likely
       track for the animal by checking step-1 locations
    Params:
        None
    Returns:
        None
        * Manipulates track objects directly
    """
    def avoid_overlapping_assignment(self):
        for track_id in range(self.n_obj):
            other_tracks = list(range(self.n_obj))
            other_tracks.pop(track_id)
            track_point_dict = self.get_track(track_id).point_dict

            for key in track_point_dict:
                for other_track_id in other_tracks:
                    if track_point_dict[key]['point'] == self.get_track(other_track_id).point_dict[key]['point']:
                        initial_minus_one_point = track_point_dict[str(int(key) - 1)]['point']
                        other_minus_one_point = self.get_track(other_track_id).point_dict[str(int(key) - 1)]['point']
                        initial_point = track_point_dict[key]['point']

                        if initial_minus_one_point is not None and \
                                other_minus_one_point is not None and initial_point is not None:
                            distance_initial_track = distance.euclidean(initial_minus_one_point, initial_point)
                            distance_other_track = distance.euclidean(other_minus_one_point, initial_point)
                            if distance_initial_track > distance_other_track:
                                self.get_track(track_id).add_point(key, track_point_dict[str(int(key) - 1)]['point'])
                            else:
                                self.get_track(other_track_id).add_point(key,
                                                                         self.get_track(other_track_id).point_dict[
                                                                             str(int(key) - 1)]['point'])

    # Filter volatile track movements and separate close tracks
    """
    Computes: 
        Avoids the rapid shifts from the tracks that are unnatural in sense. Secondly, separate_overlapping_tracks()
        is used to avoid assignment of tracks within perimeter of other tracks. This computation is used for last 
        filtration of tracks proposed
    Params:
        None
    Returns:
        None
        * Manipulates track objects directly
    """
    def last_stabilization_filter(self):
        # avoid swapping
        for track_id in range(self.n_obj):
            track_point_dict = self.get_track(track_id).point_dict
            prev_key = None
            for key in track_point_dict:
                if key != str(0):
                    temp_distance = distance.euclidean(track_point_dict[str(int(key) - 1)]['point'],
                                                       track_point_dict[key]['point'])
                    if temp_distance > 50:
                        if prev_key is not None:
                            temp_distance_2 = distance.euclidean(track_point_dict[str(prev_key)]['point'],
                                                                 track_point_dict[key]['point'])
                            if temp_distance_2 < 10:
                                for idx in range(prev_key, int(key)):
                                    self.get_track(track_id).add_point(idx, track_point_dict[str(prev_key)]['point'])
                                    self.get_track(track_id).add_confidence(idx, track_point_dict[str(prev_key)][
                                        'confidence'])
                        prev_key = int(key) - 1

        # separate overlapping tracks
        self.separate_overlapping_tracks()

    # Separation of overlapping tracks due to wrong classification
    """
    Computes: 
       Separate the tracks that are too close to each other due to errors induced by filling functions. It is used as a
       last resort to correct and stabilize the tracks.
    Params:
        verbose: Boolean for printing debug information
    Returns:
        None
    """
    def separate_overlapping_tracks(self, verbose=False):
        for track_id in range(self.n_obj):
            track_point_dict = self.get_track(track_id).point_dict

            overlap_count = 0
            temp_other_track = None
            for key in track_point_dict:
                overlap_bool = False
                for other_track_id in range(self.n_obj)[track_id + 1:]:
                    if int(key) != 0:
                        initial_minus_one_point = track_point_dict[str(int(key) - 1)]['point']
                        other_minus_one_point = self.get_track(other_track_id).point_dict[str(int(key) - 1)]['point']
                        initial_point = track_point_dict[key]['point']
                        other_point = self.get_track(other_track_id).point_dict[key]['point']

                        if initial_minus_one_point is not None and other_minus_one_point is not None \
                                and initial_point is not None and other_point is not None:
                            distance_overlap = distance.euclidean(initial_point, other_point)
                            if distance_overlap < 10:
                                overlap_count += 1
                                overlap_bool = True
                                temp_other_track = other_track_id
                if overlap_count > 0 and not overlap_bool:
                    avg_point = ((track_point_dict[str(int(key) - overlap_count)]['point'][0] +
                                  self.get_track(temp_other_track).point_dict[str(int(key) - overlap_count)]['point'][
                                      0])
                                 / 2,
                                 (track_point_dict[str(int(key) - overlap_count)]['point'][1] +
                                  self.get_track(temp_other_track).point_dict[str(int(key) - overlap_count)]['point'][
                                      1])
                                 / 2)
                    avg_dist_p = ((distance.euclidean(avg_point,
                                                      track_point_dict[str(int(key) - overlap_count - 1)]['point']) +
                                   distance.euclidean(avg_point,
                                                      track_point_dict[str(int(key) + 1)]['point'])) / 2)
                    avg_dist_o = ((distance.euclidean(avg_point, self.get_track(temp_other_track).point_dict[
                        str(int(key) - overlap_count - 1)]['point']) + distance.euclidean(
                        avg_point, self.get_track(temp_other_track).point_dict[str(int(key) + 1)]['point'])) / 2)

                    p_track_b = self.find_nearest_certain_point(track_id, int(key) - overlap_count, 'backward')
                    o_track_b = self.find_nearest_certain_point(temp_other_track, int(key) - overlap_count, 'backward')
                    p_track_f = self.find_nearest_certain_point(track_id, int(key), 'forward')
                    o_track_f = self.find_nearest_certain_point(temp_other_track, int(key), 'forward')

                    if verbose:
                        print('Overlapping between ', int(key) - overlap_count, 'and', key)
                        print('Prev_Location of Track (', track_id, '): ',
                              track_point_dict[str(int(key) - overlap_count - 1)]['point'])
                        print('Prev_Location of Track (', temp_other_track, '): ',
                              self.get_track(temp_other_track).point_dict[str(int(key) - overlap_count - 1)]['point'])
                        print('Location of Track (', track_id, '): ',
                              track_point_dict[str(int(key) - overlap_count)]['point'])
                        print('Location of Track (', temp_other_track, '): ',
                              self.get_track(temp_other_track).point_dict[str(int(key) - overlap_count)]['point'])
                        print('Next_Location of Track (', track_id, '): ',
                              track_point_dict[str(int(key) + 1)]['point'])
                        print('Next_Location of Track (', temp_other_track, '): ',
                              self.get_track(temp_other_track).point_dict[str(int(key) + 1)]['point'])
                        print('----------')
                        print(avg_dist_p)
                        print(avg_dist_o)
                        print('----------')
                        print(p_track_b, o_track_b, p_track_f, o_track_f)
                        print(track_point_dict[str(int(key) - overlap_count)]['confidence'])
                        print('----------')

                    if avg_dist_p > avg_dist_o:
                        self.fix_assignment_of_tracks(track_id,
                                                      int(key) - overlap_count - p_track_b,
                                                      int(key) + p_track_f)
                    else:
                        self.fix_assignment_of_tracks(temp_other_track,
                                                      int(key) - overlap_count - o_track_b,
                                                      int(key) + o_track_f)

                    overlap_count = 0

    # Helper functions for separation of identified tracks
    """
    Computes: 
       Find the nearest certain point to assign while separating the tracks. This is a helper function to be used
       in the separate_overlapping_tracks()
    Params:
        track_id: ID of the track corresponding with the doe_id. Integer with domain of 0 to self.n_obj-1
        key: frame number. Integer with domain of 0 to len(self.dict)
        method: search method from given frame number (key). String with options: 'forward' and 'backward'
    Returns:
        c: count of frames to reach certain assignment of a given track (int)
    """
    def find_nearest_certain_point(self, track_id, key, method):
        c = 1
        track_point_dict = self.get_track(track_id).point_dict
        if method == 'backward':
            while track_point_dict[str(int(key) - c)]['confidence'] < 0.8:
                c += 1
            return c
        if method == 'forward':
            while track_point_dict[str(int(key) + c)]['confidence'] < 0.8:
                c += 1
            return c

    # Helper functions for separation of tracks by manipulation of tracks
    """
    Computes: 
        Helper function to fix the assignment of tracks while using the separation function. It corrects the wrong 
        assigned track and called within separate_overlapping_tracks().
    Params:
        track_id: ID of the track corresponding with the doe_id. Integer with domain of 0 to self.n_obj-1
        start_frame: starting frame number to fix the track. Integer with domain of 0 to len(self.dict)-1
        end_frame: ending frame number to fix the track. Integer with domain of 0 to len(self.dict)-1
        verbose: Boolean for printing debug information
    Returns:
        None
        * Manipulates track objects directly
    """
    def fix_assignment_of_tracks(self, track_id, start_frame, end_frame, verbose=False):
        track_point_dict = self.get_track(track_id).point_dict
        ext_arr = self.get_extrapolation_between_points(track_point_dict[str(start_frame - 1)]['point'],
                                                        track_point_dict[str(end_frame + 1)]['point'],
                                                        end_frame - start_frame)
        for idx in range(start_frame, end_frame):
            self.get_track(track_id).add_point(idx, ext_arr[idx - start_frame])
            self.get_track(track_id).add_confidence(idx, 0.8)

        if verbose:
            print('++++++++++++++')
            track_point_dict = self.get_track(track_id).point_dict
            for idx in range(start_frame - 1, end_frame + 1):
                print(track_point_dict[str(idx)])
            print('++++++++++++++')

    # Main function to track rabbits
    """
    Computes: 
        Main generative function for tracks after a class object creation given the threshold for classifications to
        build assign initial points. It calls above declared functions sequentially.
    Params:
        threshold: Threshold to accept probabilities from the classification model to assign animals within a track.
                   Float number with domain of 0 to 1.
    Returns:
        None
    """
    def generate_tracks(self, threshold):
        # initialize tracks
        for track_id in range(self.n_obj):
            self.initialize_track(track_id, threshold)

        # clean the flashy occurrences of animals (Uncertain moments)
        self.clean_in_between_occurrences(n_frames=5)

        # identify and inject silver points then clean single injections
        self.fix_silver_key_points()
        self.clean_in_between_occurrences(n_frames=1)

        # fill points in between using custom algorithm (refer function definition)
        for track_id in range(self.n_obj):
            self.fill_in_between_classified_points(track_id)

        # fill initial points
        self._fill_initial_points()

        # fill final points
        self._fill_final_points()

        # stabilize tracks and avoid overlapping assignments
        self.stabilize_tracks(window=25, dist_threshold=50)
        self.avoid_overlapping_assignment()

        # final layer of stabilization of tracks
        self.last_stabilization_filter()
