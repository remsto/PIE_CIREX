"Drone that goes along the walls. If it detects a body, it will grasp it"

import random
import numpy as np
import cv2


from math import atan2, sqrt, atan, cos, pi
from typing import Optional
from enum import Enum, auto

from spg_overlay.drone_abstract import DroneAbstract
from spg_overlay.drone_sensors import DroneSemanticCones
from spg_overlay.utils import normalize_angle

from solutions.algorithme_astar import astar

from .KHT import *


def rot(angle):
    return np.matrix([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])


class MyRescueDrone(DroneAbstract):
    class Activity(Enum):
        """
        Possible states of the drone
        """
        STARTING = auto()
        STAND_BY = auto()
        GO_GETEM = auto()
        BRING_TO_RESCUE = auto()
        GO_TO_SLEEP = auto()

    def __init__(self, last_right_state: bool = None, identifier: Optional[int] = None, **kwargs):
        super().__init__(identifier=identifier,
                         should_display_lidar=False,
                         **kwargs)

        # State of the drone from the Enum Activity class
        self.state = self.Activity.STARTING

        # Constants for the drone
        self.last_right_state = last_right_state
        self.base_speed = 0.1
        self.base_rot_speed = 0.1
        self.epsilon = 5
        self.scale = 10
        self.path = []
        self.reverse_path = []
        self.debug_id = self.identifier
        self.l_pos = []
        self.l_vit = []
        self.l_true_pos = []
        self.raw_data = []
        self.pos2 = []
        self.time = []
        self.accumulator_map = np.zeros(
            (round(self.size_area[1] // self.scale) + 1, round(self.size_area[0] // 5) + 1))

        self.map = np.zeros(self.size_area)
        self.explored_map = np.ones((
            self.size_area[1]//self.scale, self.size_area[0]//self.scale))
        self.HSpace = HoughSpace([], self.size_area, 200, 200)
        self.nstep = 0
        self.last_20_pos = [(i, i) for i in range(200)]
        self.turning_time = 0

        self.lidar_angles = self.lidar().ray_angles
        self.nb_angles = len(self.lidar_angles)

        self.rescue_HSpace = HoughSpace([], self.size_area, 200, 200)
        self.found_rescue_center = False

    def define_message(self):
        """
        Here, we don't need communication...
        """
        pass

    def process_communication_sensor(self):

        try:
           x, y =self.get_pos()
           pos = np.array([x, y])
        except:
            pass

        if self.communication:
            messages = self.communication.received_message
            for msg in messages:
                try:
                    other_pos = msg[2]
                    if np.sqrt(np.dot(pos - other_pos, pos - other_pos)) < 200:
                        return False

                except Exception:
                    pass
            
        return True

    def path_to_points(self, path, map):
        """ Transforms a discrete path of doubles into a set of discrete points"""
        points = []
        direction = (path[1][0]-path[0][0], path[1][1]-path[0][1])
        for i in range(2, len(path)):
            if (path[i][0]-path[i-1][0], path[i][1]-path[i-1][1]) != direction:
                direction = (path[i][0]-path[i-1][0], path[i][1]-path[i-1][1])
                path[i-1] = (path[i-1][0] + (map[path[i-1][0]-1][path[i-1][1]] - map[path[i-1][0]+1][path[i-1][1]])*2,
                             path[i-1][1] + (map[path[i-1][0]][path[i-1][1]-1] - map[path[i-1][0]][path[i-1][1]+1])*2)
                points.append(path[i-1])
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

    def filter_position(self):

        cur_pos = self.measured_position()
        if (len(self.time) >= 2):
            self.time.append(self.time[-1]+1)
            d_vit = self.measured_velocity()

            d_pos = 0.5*(np.array([[d_vit[0]], [d_vit[1]]])+self.l_vit[-1])

            k_pos = 0.006
            pos1 = self.l_pos[-1]+d_pos  # rot.dot(d_pos)
            pos2 = np.array([[cur_pos[0]], [cur_pos[1]]])
            mix = k_pos*pos2 + (1-k_pos)*pos1
            self.l_pos.append(mix)

            self.raw_data.append(cur_pos)

            self.l_vit.append(np.array([[d_vit[0]], [d_vit[1]]]))
        else:
            self.time.append(0)
            d_vit = self.measured_velocity()
            self.l_pos.append(np.array([[cur_pos[0]], [cur_pos[1]]]))
            self.raw_data.append(cur_pos)
            self.l_vit.append(np.array([[d_vit[0]], [d_vit[1]]]))

            self.time.append(1)

    def get_pos(self):
        return self.l_pos[-1][0][0], self.l_pos[-1][1][0]

    def is_stucked(self, last_20_pos):
        var = np.var(last_20_pos, axis=0)
        return var[0] <= 2 and var[1] <= 2

    def update_last_20_pos(self, last_20_pos, new_pos):
        last_20_pos.pop(0)
        last_20_pos.append(new_pos)

    def get_rescue_center(self):
        points = []

        drone_pos = self.l_pos[-1]
        drone_angle = self.measured_angle()

        for r in self.semantic_cones().sensor_values:
            if r.entity_type == DroneSemanticCones.TypeEntity.RESCUE_CENTER:
                obstacle_pos = r.distance * np.dot(rot(drone_angle + r.angle), np.array([1, 0]).reshape(2, 1)) + drone_pos
                x_cor = max(min(self.size_area[0] - 1, obstacle_pos[0, 0]), 0)
                y_cor = max(min(self.size_area[1] - 1, obstacle_pos[1, 0]), 0)
                points.append([x_cor, y_cor])

        self.rescue_HSpace.add_points_to_process(points)

    def set_anchor_pos(self):
        self.rescue_HSpace.point_transform()
        lines = self.rescue_HSpace.compute_lines_length()
        tx, ty = self.get_pos()
        pos = np.array([tx, ty])

        if not len(lines):
            return False

        else:
            l = lines[0]
            x = int((l.p1[0] + l.p2[0]) // 2)
            y = int((l.p1[1] + l.p2[1]) // 2)

        self.rescue_center_pos = np.array([x, y])

        x_sb, y_sb = x, y
        DISTANCE_FROM_CENTER = 50

        if l.orientation:   # Droite verticale
            if tx > x:
                x_sb += DISTANCE_FROM_CENTER
            else:
                x_sb -= DISTANCE_FROM_CENTER

        else:
            if ty > y:
                y_sb += DISTANCE_FROM_CENTER
            else:
                y_sb -= DISTANCE_FROM_CENTER
        
        self.stand_by_pos = np.array([x_sb, y_sb])

        return True

    def control(self):
        print(self.state)
        self.filter_position()

        command = {self.longitudinal_force: 0.0,
                   self.lateral_force: 0.0,
                   self.rotation_velocity: 0.0,
                   self.grasp: 0}

        x, y = self.get_pos()
        pos = np.array([x, y])
        x = min(int(x), self.size_area[0])//self.scale
        y = min(int(y), self.size_area[1])//self.scale
        self.update_last_20_pos(self.last_20_pos, (y, x))

        if self.state == self.Activity.STARTING:
            if self.nstep < 30:
                self.get_rescue_center()

            elif not self.found_rescue_center:
                self.get_rescue_center()
                self.found_rescue_center = self.set_anchor_pos()

                if self.found_rescue_center:
                    self.state = self.Activity.GO_TO_SLEEP

            command[self.rotation_velocity] = 0.5

        elif self.state == self.Activity.STAND_BY:
            if self.process_communication_sensor() and (self.process_semantic_sensor_wounded(self.semantic_cones()) != (0.0, 0.0)):
                self.state = self.Activity.GO_GETEM

            elif np.sqrt(np.dot(pos - self.stand_by_pos, pos - self.stand_by_pos)) > 40:
                self.state = self.Activity.GO_TO_SLEEP

        elif self.state is self.Activity.GO_GETEM:
            wounded_person_angle, wounded_person_distance = self.process_semantic_sensor_wounded(
                self.semantic_cones())
            if wounded_person_angle < 0:
                command[self.rotation_velocity] = -0.5
                command[self.longitudinal_force] = 0.5
            elif wounded_person_angle > 0:
                command[self.rotation_velocity] = 0.5
                command[self.longitudinal_force] = 0.5

            if wounded_person_distance < 20 and wounded_person_distance > 0:
                command[self.grasp] = 1
                self.state = self.Activity.BRING_TO_RESCUE

        elif self.state is self.Activity.BRING_TO_RESCUE:
            command[self.grasp] = 1
            x, y = self.get_pos()

            pos = np.array([x, y])

            x_diff = self.rescue_center_pos[0] - x
            y_diff = self.rescue_center_pos[1] - y

            alpha = atan2(y_diff, x_diff)
            alpha = normalize_angle(alpha)
            a2 = normalize_angle(self.measured_angle())
            alpha_diff = normalize_angle(alpha-a2)

            rescue_center_angle, rescue_center_distance = alpha_diff, np.sqrt(np.dot(pos - self.rescue_center_pos, pos - self.rescue_center_pos))

            if rescue_center_angle < 0:
                command[self.rotation_velocity] = -0.5
                command[self.longitudinal_force] = 0.5
            elif rescue_center_angle > 0:
                command[self.rotation_velocity] = 0.5
                command[self.longitudinal_force] = 0.5

            if rescue_center_distance < 20 and rescue_center_distance > 0:
                command[self.grasp] = 1
                self.state = self.Activity.GO_TO_SLEEP
        
        elif self.state is self.Activity.GO_TO_SLEEP:
            x, y = self.get_pos()

            pos = np.array([x, y])

            x_diff = self.stand_by_pos[0] - x
            y_diff = self.stand_by_pos[1] - y

            alpha = atan2(y_diff, x_diff)
            alpha = normalize_angle(alpha)
            a2 = normalize_angle(self.measured_angle())
            alpha_diff = normalize_angle(alpha-a2)

            stand_by_angle, stand_by_distance = alpha_diff, np.sqrt(np.dot(pos - self.stand_by_pos, pos - self.stand_by_pos))

            if stand_by_angle < 0:
                command[self.rotation_velocity] = -0.5
                command[self.longitudinal_force] = 0.5
            elif stand_by_angle > 0:
                command[self.rotation_velocity] = 0.5
                command[self.longitudinal_force] = 0.5

            if stand_by_distance < 20 and stand_by_distance > 0:
                self.state = self.Activity.STAND_BY


        touch_value = self.process_touch_sensor()
        if touch_value == "left":
            command[self.lateral_force] = self.base_speed
        elif touch_value == "right":
            command[self.lateral_force] = -self.base_speed

        self.nstep += 1

        return command
