"Drone that goes along the walls. If it detects a body, it will grasp it"

import random
import numpy as np


from math import cos, pi
from typing import Optional
from enum import Enum

from spg_overlay.drone_abstract import DroneAbstract
from spg_overlay.drone_sensors import DroneSemanticCones
from spg_overlay.utils import normalize_angle


class MyWallDrone(DroneAbstract):
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
        self.base_speed = 0.1
        self.base_rot_speed = 0.1
        self.epsilon = 5

        # A counter that allow the drone to turn for a specified amount of time
        self.turning_time = 0

        self.lidar_angles = self.lidar().ray_angles
        self.nb_angles = len(self.lidar_angles)

    def turn_90(self, direction):
        """
        Stop the drone and turn 90 degress toward the given direction
        """
        command = {self.longitudinal_force: 0.0,
                   self.lateral_force: 0.0,
                   self.rotation_velocity: 0.0}

        # Initialize turning state
        if self.turning_time == 0:
            if direction == "left":
                self.state = self.Activity.TURNING_LEFT
            elif direction == "right":
                self.state = self.Activity.TURNING_RIGHT
            else:
                raise ValueError(
                    "Bad direction for the drone (Must be left or right)")

        if direction == "left":
            command[self.rotation_velocity] = -0.1

        elif direction == "right":
            command[self.rotation_velocity] = 0.1
        else:
            raise ValueError(
                "Bad direction for the drone (Must be left or right)")

        self.turning_time += 1

        # End turning state
        if self.turning_time == 60:
            self.state = self.Activity.SEARCHING
            self.turning_time = 0
        return command

    def define_message(self):
        """
        Here, we don't need communication...
        """
        pass

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


    def corridor_mode(self):
        """
        Define how the drone has to move when in a corridor
        """
        limit_distance = 25
        lidar_values = self.process_lidar_sensor(self.lidar())
        wall_right = lidar_values[89] < limit_distance
        wall_left = lidar_values[0] < limit_distance
        wall_front = lidar_values[44] < limit_distance
        if wall_left and wall_right and not wall_front:
            # AVANCER
            pass
        elif wall_left:
            pass

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
        # print(self.lidar_angles)
        # print(values)
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

            # First prototype of Saving
            # wounded_person_angle, wounded_person_distance = self.process_semantic_sensor_wounded(
            #     self.semantic_cones())
            # if wounded_person_angle != 0:
            #     self.state = self.Activity.FOUND

        elif self.state is self.Activity.FOUND:
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

            # End of first prototype of saving
        elif self.state is self.Activity.GOING_BACK:

            command[self.grasp] = 1
            wall_front = values[front_index] < distance_max
            wall_front_right = values[frontright_index] < distance_max*3
            wall_front_left = values[frontleft_index] < distance_max*3
            wall_left = values[left_index] < distance_max
            if not wall_front and not wall_left:
                command[self.longitudinal_force] = self.base_speed
            elif wall_front_right and wall_front_left:
                command[self.rotation_velocity] = self.base_rot_speed
            else:
                x = values[left_index]
                alpha = self.lidar_angles[backleft_index] + pi/2
                y = values[backleft_index]
                if y > x/cos(alpha) + self.epsilon:
                    command[self.rotation_velocity] = self.base_rot_speed
                elif y < x/cos(alpha) - self.epsilon:
                    command[self.rotation_velocity] = -self.base_rot_speed
                else:
                    command[self.longitudinal_force] = self.base_speed
            center_angle, center_distance = self.process_semantic_sensor_center(
                self.semantic_cones())
            if center_distance < distance_max and center_distance > 0:
                self.state = self.Activity.SEARCHING

        touch_value = self.process_touch_sensor()
        if touch_value == "left":
            command[self.lateral_force] = self.base_speed
        elif touch_value == "right":
            command[self.lateral_force] = -self.base_speed

        """
        # First prototype of control
        against_a_wall = values[89] < distance_max and values[46] > distance_max

        if against_a_wall:
            command[self.longitudinal_force] += self.base_speed
        elif values[46] < distance_max:
            command[self.longitudinal_force] += -self.base_speed
            command[self.rotation_velocity] += -self.base_rot_speed
        elif self.last_right_state:
            command[self.longitudinal_force] += -self.base_speed
            command[self.rotation_velocity] += self.base_rot_speed
        else:
            command[self.longitudinal_force] += self.base_speed

        self.last_right_state = against_a_wall
        # End of First Prototype of control
        """

        if self.state is self.Activity.TURNING_LEFT:
            command = self.turn_90("left")

        if self.state is self.Activity.TURNING_RIGHT:
            command = self.turn_90("right")

        return command
