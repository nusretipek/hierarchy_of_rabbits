import json
from scipy.spatial import distance


class Track:

    # init function
    def __init__(self, track_id):
        self.track_id = track_id
        self.doe_id = None
        self.prev_points = []
        self.prev_points_confidence = []

    # get functions
    def get_last_point(self):
        if len(self.prev_points) > 0:
            return self.prev_points[-1]
        else:
            return None

    # get functions
    def get_doe_id(self):
        return self.doe_id

    def get_all_points(self):
        return self.prev_points

    def get_last_confidence(self):
        if len(self.prev_points_confidence) > 0:
            return self.prev_points_confidence[-1]
        else:
            return None

    def get_all_confidence(self):
        return self.prev_points_confidence

    # insert functions
    def add_point(self, point_tuple):
        self.prev_points.append(point_tuple)

    def add_confidence(self, p):
        self.prev_points_confidence.append(p)

    def assign_doe_to_track(self, doe_id):
        self.doe_id = doe_id


class DistanceTracker:

    # init function
    def __init__(self, pred_dict, n_obj=4):
        self.dict = pred_dict
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
    def get_centroid_bbox(bbox_arr):
        centroid_arr = []
        for bbox in bbox_arr:
            centroid_arr.append((int(bbox[0] + bbox[2]/2), int(bbox[1] + bbox[3]/2)))
        return centroid_arr

    def get_track(self, track_id):
        return self.tracks[track_id].get_all_points()

    def get_track_doe_id(self, track_id):
        return self.tracks[track_id].get_doe_id()

    # calculate Euclidian distance between points

    @staticmethod
    def calculate_euclidian_distance(tracks, centroids):
        distance_dict = {}
        for track in tracks:
            counter = 0
            distance_dict[track.track_id] = {}
            for centroid in centroids:
                distance_dict[track.track_id][counter] = distance.euclidean(track.get_last_point()[0], centroid)
                counter += 1
        return distance_dict

    @staticmethod
    def get_minimum_distance(distance_dict):
        minimum_distance = None
        minimum_distance_track = None
        minimum_distance_centroid = None
        for key in distance_dict:
            for element in distance_dict[key]:
                if minimum_distance is None or distance_dict[key][element] < minimum_distance:
                    minimum_distance = distance_dict[key][element]
                    minimum_distance_track = key
                    minimum_distance_centroid = element
        return minimum_distance_track, minimum_distance_centroid, minimum_distance

    @staticmethod
    def delete_track_centroid(distance_dict, key, element):
        del distance_dict[key]
        for key in distance_dict:
            del distance_dict[key][element]
        return distance_dict

    # generate tracks

    def initialize_tracks(self):
        counter = 0
        for centroid in self.get_centroid_bbox(self.get_bbox(0)):
            self.tracks[counter].add_point((centroid, 1))
            if self.get_classification(0)[counter][1] > 0.98:
                self.tracks[counter].assign_doe_to_track(self.get_classification(0)[counter][0])
            counter += 1

        while counter < self.n_obj:
            self.tracks[counter].add_point(((-1, -1), 0))
            counter += 1

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
