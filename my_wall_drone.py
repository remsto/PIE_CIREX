"Drone that goes along the walls"

import random
import math
import numpy as np

from typing import Optional

from spg_overlay.drone_abstract import DroneAbstract
from spg_overlay.utils import normalize_angle


class MyWallDrone(DroneAbstract):
    def __init__(self, identifier: Optional[int] = None, **kwargs):
        super().__init__(identifier=identifier,
                         should_display_lidar=False,
                         **kwargs)
        self.counterStraight = 0
        self.angleStopTurning = 0
        self.isTurning = False

    def define_message(self):
        """
        Here, we don't need communication...
        """
        pass

    def process_touch_sensor(self):
        """
        Returns True if the drone hits an obstacle
        """
        touched = False
        detection = max(self.touch().sensor_values)

        if detection > 0.5:
            touched = True

        return touched

    def process_lidar_sensor(self, the_lidar_sensor):
        command = {self.longitudinal_force: 0.0,
                   self.lateral_force: 0.0,
                   self.rotation_velocity: 0.0}
        rotation_velocity = 0.6

        values = the_lidar_sensor.get_sensor_values()

        if values[89] < 50 and values[46] > 50:
            command[self.longitudinal_force] += 0.5
        elif values[46] < 50:
            command[self.rotation_velocity] += -rotation_velocity
        else:
            command[self.longitudinal_force] += 0.5

        return command

    def show_values(self):
        print("ray_angles : ", self.lidar().ray_angles)

        return None

    def control(self):
        """
        The Drone will move forward and turn for a random angle when an obstacle is hit
        """
        return self.process_lidar_sensor(self.lidar())
