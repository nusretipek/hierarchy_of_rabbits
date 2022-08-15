import os


class ArgumentError(ValueError):
    pass


class RabbitDirectory:
    def __init__(self, location, save_path):
        if os.path.exists(location):
            self.location = location
            self.cameraText = location.rsplit('\\', 1)[1]
        else:
            raise ArgumentError("Location does not exist!")
        self.savePath = os.path.join(save_path, self.cameraText)

    def make_directory(self):
        if not os.path.exists(self.savePath):
            os.mkdir(self.savePath)


class RabbitFile:
    def __init__(self, location):
        self.location = location
        self.name = location.rsplit('\\', 1)[1].rsplit('.', 1)[0]

    def get_time(self):
        return self.name[6:17]


if __name__ == '__main__':
    temp_dir = RabbitDirectory("D:\\Rabbit Research Videos\\HPC_Analysis\\WP32_Cycle3\\Action_Diagrams\\Camera 12",
                               'D:\\Rabbit Research Videos\\HPC_Analysis\\Test')
    temp_dir.make_directory()
