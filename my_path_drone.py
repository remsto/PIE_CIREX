"Test class for a drone that follows a given path"

import random
import numpy as np
import sys


from math import atan, pi
from typing import Optional
from enum import Enum

from spg_overlay.drone_abstract import DroneAbstract
from spg_overlay.drone_sensors import DroneSemanticCones
from spg_overlay.utils import normalize_angle

from solutions.algorithme_astar import astar


class MyPathDrone(DroneAbstract):
    class Activity(Enum):
        """
        Possible states of the drone
        """
        SEARCHING = 1
        FOUND = 2
        GOING_BACK = 3
        TURNING_RIGHT = 4
        TURNING_LEFT = 5

    def __init__(self, last_right_state: bool = None, identifier: Optional[int] = None, **kwargs):
        super().__init__(identifier=identifier,
                         should_display_lidar=False,
                         **kwargs)

        # State of the drone from the Enum Activity class
        self.state = self.Activity.SEARCHING

        # Constants for the drone
        self.last_right_state = last_right_state
        self.base_speed = 0.5
        self.base_rot_speed = 0.1
        self.epsilon = 5

        # A counter that allow the drone to turn for a specified amount of time
        self.turning_time = 0

        self.lidar_angles = self.lidar().ray_angles
        self.path = []
        self.map = [[0 for _ in range(1112//5)] for _ in range(750//5)]
        for i in range(200):
            self.map[119][i] = 1

    def define_message(self):
        """
        Here, we don't need communication...
        """
        pass

    def path_to_points(self, path):
        """ Transforms a discrete path of doubles into a set of discrete points"""
        points = []
        direction = (path[1][0]-path[0][0], path[1][1]-path[0][1])
        for i in range(2, len(path)):
            if (path[i][0]-path[i-1][0], path[i][1]-path[i-1][1]) != direction:
                points.append(path[i-1])
                direction = (path[i][0]-path[i-1][0], path[i][1]-path[i-1][1])
        points.append(path[-1])
        return points

    def control(self):
        command = {self.longitudinal_force: 0.1,
                   self.lateral_force: 0.0,
                   self.rotation_velocity: 0.0,
                   self.grasp: 0}

        if self.state is self.Activity.SEARCHING:
            x, y = self.position
            x = min(int(x), 1112)//5
            y = min(int(y), 750)//5
            print("test1")
            print(y, x)
            self.path = self.path_to_points(
                astar(self.map, (y, x), (100, 170)))
            print("test2")
            self.state = self.Activity.FOUND
        elif self.state is self.Activity.FOUND:
            if self.path == []:
                self.state = self.Activity.SEARCHING
            else:
                destination = self.path[0]
                x, y = self.position
                x = min(int(x), 1112)//5
                y = min(int(y), 750)//5
                x_diff = destination[1]-x
                y_diff = destination[0]-y
                alpha = 0
                if x_diff > 0:
                    alpha = atan(y_diff/x_diff) % (2*pi)
                elif x_diff < 0:
                    alpha = pi + atan(y_diff/x_diff)
                else:
                    alpha = 0

                alpha_diff = (alpha-self.angle) % (2*pi)
                if alpha_diff < pi:
                    command[self.rotation_velocity] += 0.2
                elif alpha_diff > 0:
                    command[self.rotation_velocity] -= 0.2
                if abs(x_diff) <= 1 and abs(y_diff) <= 1:
                    self.path.pop(0)
                print(self.path)
                print(y, x)

        return command
