"Drone that goes along the walls. If it detects a body, it will grasp it"

import random
import numpy as np


from math import atan, cos, pi
from typing import Optional
from enum import Enum

from spg_overlay.drone_abstract import DroneAbstract
from spg_overlay.drone_sensors import DroneSemanticCones
from spg_overlay.utils import normalize_angle

from solutions.algorithme_astar import astar


class MyWallPathDrone(DroneAbstract):
    class Activity(Enum):
        """
        Possible states of the drone
        """
        SEARCHING = 1
        FOUND_WOUNDED = 2
        GOING_BACK = 3
        FOUND_CENTER = 4
        TURNING_RIGHT = 5
        TURNING_LEFT = 6

    def __init__(self, last_right_state: bool = None, identifier: Optional[int] = None, **kwargs):
        super().__init__(identifier=identifier,
                         should_display_lidar=False,
                         **kwargs)

        # State of the drone from the Enum Activity class
        self.state = self.Activity.SEARCHING

        # Constants for the drone
        self.last_right_state = last_right_state
        self.base_speed = 0.2
        self.base_rot_speed = 0.1
        self.epsilon = 5
        self.scale = 5
        self.base_pos = (130, 50)
        self.path = []

        # Dummy map for tests
        self.map = [[0 for _ in range(1112//5)] for _ in range(750//5)]
        for i in range(200):
            self.map[119][i] = 1

        # A counter that allow the drone to turn for a specified amount of time
        self.turning_time = 0

        self.lidar_angles = self.lidar().ray_angles
        self.nb_angles = len(self.lidar_angles)

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

    def process_touch_sensor(self):
        """
        Returns True if the drone hits an obstacle
        """
        touch_sensor = self.touch()
        touch_angles = touch_sensor._ray_angles
        touch_values = touch_sensor.sensor_values
        if touch_values[3] > 0.7:
            return "left"
        elif touch_values[8] > 0.7:
            return "right"
        return None

    def process_semantic_sensor_wounded(self, the_semantic_sensor):
        """
        Returns the angle where a wounded person is, and 0 if not found
        """
        values = the_semantic_sensor.sensor_values

        for data in values:
            if data.entity_type == DroneSemanticCones.TypeEntity.WOUNDED_PERSON:
                return data.angle, data.distance
        return 0.0, 0.0

    def process_semantic_sensor_center(self, the_semantic_sensor):
        """
        Returns the angle where the center is, and 0 if not found
        """
        values = the_semantic_sensor.sensor_values

        for data in values:
            if data.entity_type == DroneSemanticCones.TypeEntity.RESCUE_CENTER:
                return data.angle, data.distance
        return 0.0, 0.0

    def process_lidar_sensor(self, the_lidar_sensor):
        """
        Returns the values of the lidar sensor
        """
        values = the_lidar_sensor.get_sensor_values()
        return values

    def control(self):
        backleft_index = 11
        left_index = 22
        frontleft_index = 33
        front_index = 44
        frontright_index = 55
        right_index = 67
        backright_index = 77

        command = {self.longitudinal_force: 0.0,
                   self.lateral_force: 0.0,
                   self.rotation_velocity: 0.0,
                   self.grasp: 0}
        distance_max = 20

        values = self.process_lidar_sensor(self.lidar())
        if self.state is self.Activity.SEARCHING:
            wall_front = values[front_index] < distance_max
            wall_front_right = values[frontright_index] < distance_max
            wall_right = values[right_index] < distance_max
            if not wall_front and not wall_right:
                command[self.longitudinal_force] = self.base_speed
            elif wall_front:
                command[self.rotation_velocity] = -self.base_rot_speed
            else:
                x = values[right_index]
                alpha = self.lidar_angles[frontright_index]
                y = values[frontright_index]
                if y > x/cos(alpha) + self.epsilon:
                    command[self.rotation_velocity] = self.base_rot_speed
                elif y < x/cos(alpha) - self.epsilon:
                    command[self.rotation_velocity] = -self.base_rot_speed
                else:
                    command[self.longitudinal_force] = self.base_speed
            wounded_person_angle, wounded_person_distance = self.process_semantic_sensor_wounded(
                self.semantic_cones())
            if wounded_person_angle != 0:
                self.state = self.Activity.FOUND_WOUNDED

        elif self.state is self.Activity.FOUND_WOUNDED:
            wounded_person_angle, wounded_person_distance = self.process_semantic_sensor_wounded(
                self.semantic_cones())
            if wounded_person_angle < 0:
                command[self.rotation_velocity] = -self.base_rot_speed
                command[self.longitudinal_force] = 0.1
            elif wounded_person_angle > 0:
                command[self.rotation_velocity] = self.base_rot_speed
                command[self.longitudinal_force] = 0.1
            if wounded_person_distance < 20 and wounded_person_distance > 0:
                command[self.grasp] = 1
                self.state = self.Activity.GOING_BACK
        elif self.state is self.Activity.GOING_BACK:
            command[self.grasp] = 1
            if self.path == []:
                x, y = self.position
                x = min(int(x), self.size_area[0])//self.scale
                y = min(int(y), self.size_area[1])//self.scale
                self.path = self.path_to_points(
                    astar(self.map, (y, x), self.base_pos))
            else:
                command[self.longitudinal_force] = 0.1
                destination = self.path[0]
                x, y = self.position
                x = min(int(x), self.size_area[0])//self.scale
                y = min(int(y), self.size_area[1])//self.scale
                print(self.path)
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
            center_angle, center_distance = self.process_semantic_sensor_center(
                self.semantic_cones())
            if center_angle != 0:
                self.state = self.Activity.FOUND_CENTER
            if center_distance < distance_max and center_distance > 0:
                self.state = self.Activity.SEARCHING
        elif self.state is self.Activity.FOUND_CENTER:
            command[self.grasp] = 1
            center_angle, center_distance = self.process_semantic_sensor_wounded(
                self.semantic_cones())
            command[self.rotation_velocity] = self.base_rot_speed
            if center_angle < 0:
                command[self.rotation_velocity] = -self.base_rot_speed
                command[self.longitudinal_force] = 0.1
            elif center_angle > 0:
                command[self.rotation_velocity] = self.base_rot_speed
                command[self.longitudinal_force] = 0.1
            if center_distance < 20 and center_distance > 0:
                self.state = self.Activity.SEARCHING

        touch_value = self.process_touch_sensor()
        if touch_value == "left":
            command[self.lateral_force] = self.base_speed
        elif touch_value == "right":
            command[self.lateral_force] = -self.base_speed

        return command
