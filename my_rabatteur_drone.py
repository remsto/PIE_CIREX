"Drone that goes along the walls. If it detects a body, it will grasp it"

import random
import numpy as np
import cv2


from math import sqrt, atan, cos, pi
from typing import Optional
from enum import Enum, auto

from spg_overlay.drone_abstract import DroneAbstract
from spg_overlay.drone_sensors import DroneSemanticCones
from spg_overlay.utils import normalize_angle

from solutions.algorithme_astar import astar

from .KHT import *


def rot(angle):
    return np.matrix([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])


class MyWallPathDrone(DroneAbstract):
    class Activity(Enum):
        """
        Possible states of the drone
        """
        SEARCHING = auto()
        FOUND_WOUNDED = auto()
        GOING_BACK = auto()
        GOING_BACK_BACK = auto()
        FOUND_CENTER = auto()
        TURNING_RIGHT = auto()
        TURNING_LEFT = auto()
        REPOSITIONING = auto()
        SEARCHING_CENTER = auto()
        WAITING_WOUNDED = auto()
        BRINGING_BACK_WOUNDED = auto()

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
        self.scale = 10
        self.path = []
        self.reverse_path = []
        self.debug_id = self.identifier

        self.map = np.zeros(self.size_area)
        self.explored_map = np.ones((
            self.size_area[1]//self.scale, self.size_area[0]//self.scale))
        self.HSpace = HoughSpace([], self.size_area, 200, 200)
        self.nstep = 0
        self.last_20_pos = [(i, i) for i in range(200)]

        # Dummy map for tests
        # self.map = [[0 for _ in range(1112//5)] for _ in range(750//5)]
        # for i in range(200):
        #     self.map[119][i] = 1

        # A counter that allow the drone to turn for a specified amount of time
        self.turning_time = 0

        self.lidar_angles = self.lidar().ray_angles
        self.nb_angles = len(self.lidar_angles)

    def define_message(self):
        """
        Here, we don't need communication...
        """
        pass

    def path_to_points(self, path, map):
        """ Transforms a discrete path of doubles into a set of discrete points"""
        points = []
        direction = (path[1][0]-path[0][0], path[1][1]-path[0][1])
        for i in range(2, len(path)):
            if (path[i][0]-path[i-1][0], path[i][1]-path[i-1][1]) != direction:
                direction = (path[i][0]-path[i-1][0], path[i][1]-path[i-1][1])
                path[i-1] = (path[i-1][0] + map[path[i-1][0]-1][path[i-1][1]] - map[path[i-1][0]+1][path[i-1][1]],
                             path[i-1][1] + map[path[i-1][0]][path[i-1][1]-1] - map[path[i-1][0]][path[i-1][1]+1])
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
        Returns the tuple angle,distance where a wounded person is, and 0 if not found
        """
        values = the_semantic_sensor.sensor_values

        for data in values:
            if data.entity_type == DroneSemanticCones.TypeEntity.WOUNDED_PERSON:
                return data.angle, data.distance
        return 0.0, 0.0

    def process_semantic_sensor_center(self, the_semantic_sensor):
        """
        Returns the tuple angle,distance where the center is, and 0 if not found
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

    def update_map(self, lidar_sensor):
        drone_pos = np.array(self.true_position()).reshape(2, 1)
        drone_angle = self.true_angle()

        semantic_data = self.semantic_cones().sensor_values
        semantic_angle = semantic_data[0].angle
        semantic_angle_id = 0

        x, y = self.position
        x = int(x//self.scale)
        y = int(y//self.scale)
        for i in range(-100//self.scale, 100//self.scale):
            for j in range(-100//self.scale, 100//self.scale):
                if x+i >= 0 and x+i < self.size_area[0]//self.scale and y+j >= 0 and y+j < self.size_area[1]//self.scale:
                    self.explored_map[y+j][x+i] = 0

        new_points = []

        for angle, mes in zip(lidar_sensor.ray_angles, lidar_sensor.get_sensor_values()):
            while (abs(angle - semantic_angle) > (5 * math.pi / 180)) and (semantic_angle < angle) and (semantic_angle_id < (len(semantic_data) - 1)):
                semantic_angle_id += 1
                semantic_angle = semantic_data[semantic_angle_id].angle

            if mes < 290:
                obstacle_pos = mes * \
                    np.dot(rot(drone_angle + angle),
                           np.array([1, 0]).reshape(2, 1)) + drone_pos

                x_cor = max(min(self.size_area[0] - 1, obstacle_pos[0, 0]), 0)
                y_cor = max(min(self.size_area[1] - 1, obstacle_pos[1, 0]), 0)

                if (abs(angle - semantic_angle) <= (5 * math.pi / 180)):
                    if (semantic_data[semantic_angle_id].entity_type != DroneSemanticCones.TypeEntity.DRONE):
                        self.map[round(x_cor), round(y_cor)] = 1
                        new_points.append([x_cor, y_cor])
                    if (semantic_data[semantic_angle_id].entity_type == DroneSemanticCones.TypeEntity.RESCUE_CENTER and self.nstep == 0):
                        self.base_pos = (round(y_cor)//self.scale,
                                         round(x_cor)//self.scale)
                        self.base_pos = (y, x)

                else:
                    self.map[round(x_cor), round(y_cor)] = 2
                    new_points.append([x_cor, y_cor])

                try:
                    for e in range(3):
                        for f in range(3):
                            self.map[round(x_cor) + e, round(y_cor) +
                                     f] = self.map[round(x_cor), round(y_cor)]
                            self.map[round(x_cor) + e, round(y_cor) -
                                     f] = self.map[round(x_cor), round(y_cor)]
                            self.map[round(x_cor) - e, round(y_cor) +
                                     f] = self.map[round(x_cor), round(y_cor)]
                            self.map[round(x_cor) - e, round(y_cor) -
                                     f] = self.map[round(x_cor), round(y_cor)]

                except IndexError:
                    pass

        self.HSpace.add_points_to_process(new_points)

    def save_map(self):
        if (self.debug_id != self.identifier):
            return

        W, H = self.size_area
        NOIR = "0 0 0\n"
        BLANC = "255 255 255\n"
        ROUGE = "255 0 0\n"

        Color = [BLANC, NOIR, ROUGE]

        with open("cartographer_drone_map.ppm", "w") as f:
            f.write("P3\n")
            f.write(f"{W} {H}\n255\n")
            for i in range(H):
                for j in range(W):
                    f.write(Color[int(self.map[j, i])])

        img = cv2.imread(
            "/home/remsto/PIE/swarm-rescue/src/swarm_rescue/cartographer_drone_map.ppm")
        self.HSpace.background = img

        self.HSpace.point_transform()
        self.HSpace.compute_lines_length()
       # self.HSpace.draw_90deg_lines_length()

    def draw_map(self, scale):
        size = [self.size_area[1] // scale, self.size_area[0] // scale]
        map = np.zeros(size)

        for line in self.HSpace.data_walls:
            if line.orientation:    # Droite verticale
                for y in range(round(line.p2[1] // scale), round(line.p1[1] // scale) + 1):
                    map[y, round(0.5 * (line.p1[0] + line.p2[0]))//scale] = 1

            else:
                for x in range(round(line.p1[0] // scale), round(line.p2[0] // scale) + 1):
                    map[round(0.5 * (line.p1[1] + line.p2[1]))//scale, x] = 1

        return map

    def is_stucked(self, last_20_pos):
        var = np.var(last_20_pos, axis=0)
        return var[0] <= 2 and var[1] <= 2

    def update_last_20_pos(self, last_20_pos, new_pos):
        last_20_pos.pop(0)
        last_20_pos.append(new_pos)

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

        x, y = self.position
        x = min(int(x), self.size_area[0])//self.scale
        y = min(int(y), self.size_area[1])//self.scale

        if self.state is self.Activity.SEARCHING_CENTER:
            center_angle, center_distance = self.process_semantic_sensor_center()
            if center_distance !=0:
                


        if self.state is self.Activity.WAITING_WOUNDED:
            # TODO
            pass
        

        self.nstep += 1

        return command
