class Track:

    # init function
    def __init__(self, doe_id):
        self.track_id = doe_id
        self.point_dict = {}

    # get functions
    def get_track_id(self):
        return self.track_id

    def get_point(self, frame):
        return self.point_dict[str(frame)]

    def get_point_dict(self):
        return self.point_dict

    def get_point_list(self):
        # implement to get list of initial element
        pass

    def get_confidence_list(self):
        # implement to get list of secondary element
        pass

    # insert functions
    def add_point(self, frame, point_tuple):
        self.point_dict[str(frame)]['point'] = point_tuple

    def add_confidence(self, frame, prob):
        self.point_dict[str(frame)]['confidence'] = prob

    # generate dict
    def initialize_point_dict(self, frame_count):
        for idx in range(frame_count):
            self.point_dict[str(idx)] = {
                'point': None,
                'confidence': None
            }
