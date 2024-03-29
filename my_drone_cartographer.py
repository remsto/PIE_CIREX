import random
import math
from copy import deepcopy
from typing import Optional

import numpy as np
import cv2

from .KHT import *

from spg_overlay.drone_abstract import DroneAbstract
from spg_overlay.misc_data import MiscData
from spg_overlay.drone_sensors import DroneSemanticCones
from spg_overlay.utils import normalize_angle, sign

import matplotlib.image as im

def rot(angle):
        return np.matrix([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

class MyDroneCartographer(DroneAbstract):
    def __init__(self,
                 identifier: Optional[int] = None,
                 misc_data: Optional[MiscData] = None,
                 DEBUG_ID: Optional[int] = -1,
                 **kwargs):
        super().__init__(identifier=identifier,
                         misc_data=misc_data,
                         should_display_lidar=False,
                         **kwargs)

        self.map = np.zeros(self.size_area)
        self.debug_id = DEBUG_ID
        self.HSpace = HoughSpace([], self.size_area, 200, 200)
        self.nstep = 0

    def update_map(self, lidar_sensor):
        drone_pos = np.array(self.true_position()).reshape(2, 1)
        drone_angle = self.true_angle()

        semantic_data = self.semantic_cones().sensor_values
        semantic_angle = semantic_data[0].angle
        semantic_angle_id = 0

        new_points = []

        for angle, mes in zip(lidar_sensor.ray_angles, lidar_sensor.get_sensor_values()):
            while (abs(angle - semantic_angle) > (5 * math.pi / 180)) and (semantic_angle < angle) and (semantic_angle_id < (len(semantic_data) - 1)):
                semantic_angle_id += 1
                semantic_angle = semantic_data[semantic_angle_id].angle

            if mes < 290:
                obstacle_pos = mes * np.dot(rot(drone_angle + angle), np.array([1, 0]).reshape(2, 1)) + drone_pos

                x_cor = max(min(self.size_area[0] - 1, obstacle_pos[0, 0]), 0)
                y_cor = max(min(self.size_area[1] - 1, obstacle_pos[1, 0]), 0)

                if (abs(angle - semantic_angle) <= (5 * math.pi / 180)):
                    if (semantic_data[semantic_angle_id].entity_type != DroneSemanticCones.TypeEntity.DRONE):
                        self.map[round(x_cor), round(y_cor)] = 1
                        new_points.append([x_cor, y_cor])
                else:
                    self.map[round(x_cor), round(y_cor)] = 2
                    new_points.append([x_cor, y_cor])

                try:
                    for e in range(3):
                        for f in range(3):
                            self.map[round(x_cor) + e, round(y_cor) + f] = self.map[round(x_cor), round(y_cor)]
                            self.map[round(x_cor) + e, round(y_cor) - f] = self.map[round(x_cor), round(y_cor)]
                            self.map[round(x_cor) - e, round(y_cor) + f] = self.map[round(x_cor), round(y_cor)]
                            self.map[round(x_cor) - e, round(y_cor) - f] = self.map[round(x_cor), round(y_cor)]

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

        img =   cv2.imread("/home/antoine/PIE/swarm-rescue/src/swarm_rescue/cartographer_drone_map.ppm")
        self.HSpace.background = img

        self.HSpace.point_transform()
        self.HSpace.draw_90deg_lines_length()


    def define_message(self):
        """
        Define the message, the drone will send to and receive from other surrounding drones.
        """
        msg_data = (self.identifier, (self.measured_position(), self.measured_angle()))
        return msg_data

    def control(self):
        """
        In this example, we only use the lidar sensor and the communication to move the drone
        The idea is to make the drones move like a school of fish.
        The lidar will help avoid running into walls.
        The communication will allow to know the position of the drones in the vicinity, to then correct its own
        position to stay at a certain distance and have the same orientation.
        """

        # print(f"{self.measured_position()} / {self.true_position()}")

        command = {self.longitudinal_force: 0.0,
                   self.lateral_force: 0.0,
                   self.rotation_velocity: 0.0}

        command_lidar, collision_lidar = self.process_lidar_sensor(self.lidar())
        found, command_comm = self.process_communication_sensor()

        if not (self.nstep % 50):
            self.update_map(self.lidar())
            self.save_map()
        self.nstep += 1

        alpha = 0.4
        alpha_rot = 0.75

        if collision_lidar:
            alpha_rot = 0.1

        # The final command  is a combinaison of 2 commands
        command[self.longitudinal_force] = \
            alpha * command_comm[self.longitudinal_force] \
            + (1 - alpha) * command_lidar[self.longitudinal_force]
        command[self.lateral_force] = \
            alpha * command_comm[self.lateral_force] \
            + (1 - alpha) * command_lidar[self.lateral_force]
        command[self.rotation_velocity] = \
            alpha_rot * command_comm[self.rotation_velocity] \
            + (1 - alpha_rot) * command_lidar[self.rotation_velocity]

        return command

    def process_lidar_sensor(self, the_lidar_sensor):
        command = {self.longitudinal_force: 1.0,
                   self.lateral_force: 0.0,
                   self.rotation_velocity: 0.0}
        rotation_velocity = 0.6

        values = the_lidar_sensor.get_sensor_values()
        ray_angles = the_lidar_sensor.ray_angles
        size = the_lidar_sensor.size

        far_angle_raw = 0
        near_angle_raw = 0
        min_dist = 1000
        if size != 0:
            # far_angle_raw : angle with the longer distance
            far_angle_raw = ray_angles[np.argmax(values)]
            min_dist = min(values)
            # near_angle_raw : angle with the nearest distance
            near_angle_raw = ray_angles[np.argmin(values)]

        far_angle = far_angle_raw
        # If far_angle_raw is small then far_angle = 0
        if abs(far_angle) < 1 / 180 * np.pi:
            far_angle = 0.0

        near_angle = near_angle_raw
        far_angle = normalize_angle(far_angle)

        # The drone will turn toward the zone with the more space ahead
        if size != 0:
            if far_angle > 0:
                command[self.rotation_velocity] = rotation_velocity
            elif far_angle == 0:
                command[self.rotation_velocity] = 0
            else:
                command[self.rotation_velocity] = -rotation_velocity

        # If near a wall then 'collision' is True and the drone tries to turn its back to the wall
        collision = False
        if size != 0 and min_dist < 50:
            collision = True
            if near_angle > 0:
                command[self.rotation_velocity] = -rotation_velocity
            else:
                command[self.rotation_velocity] = rotation_velocity

        return command, collision

    def process_communication_sensor(self):
        found_drone = False
        command_comm = {self.longitudinal_force: 0.0,
                        self.lateral_force: 0.0,
                        self.rotation_velocity: 0.0}

        if self.communication:
            received_messages = self.communication.received_message
            nearest_drone_coordinate1 = (self.measured_position(), self.measured_angle())
            nearest_drone_coordinate2 = deepcopy(nearest_drone_coordinate1)
            (nearest_position1, nearest_angle1) = nearest_drone_coordinate1
            (nearest_position2, nearest_angle2) = nearest_drone_coordinate2

            min_dist1 = 10000
            min_dist2 = 10000
            diff_angle = 0

            # Search the two nearest drones around
            for msg in received_messages:
                message = msg[1]
                coordinate = message[1]
                (other_position, other_angle) = coordinate

                dx = other_position[0] - self.measured_position()[0]
                dy = other_position[1] - self.measured_position()[1]
                distance = math.sqrt(dx ** 2 + dy ** 2)

                # if another drone is near
                if distance < min_dist1:
                    min_dist2 = min_dist1
                    min_dist1 = distance
                    nearest_drone_coordinate2 = nearest_drone_coordinate1
                    nearest_drone_coordinate1 = coordinate
                    found_drone = True
                elif distance < min_dist2 and distance != min_dist1:
                    min_dist2 = distance
                    nearest_drone_coordinate2 = coordinate

            if not found_drone:
                return found_drone, command_comm

            # If we found at least 2 drones
            if found_drone and len(received_messages) >= 2:
                (nearest_position1, nearest_angle1) = nearest_drone_coordinate1
                (nearest_position2, nearest_angle2) = nearest_drone_coordinate2
                diff_angle1 = normalize_angle(nearest_angle1 - self.measured_angle())
                diff_angle2 = normalize_angle(nearest_angle2 - self.measured_angle())
                # The mean of 2 angles can be seen as the angle of a vector, which
                # is the sum of the two unit vectors formed by the 2 angles.
                diff_angle = math.atan2(0.5 * math.sin(diff_angle1) + 0.5 * math.sin(diff_angle2),
                                        0.5 * math.cos(diff_angle1) + 0.5 * math.cos(diff_angle2))

            # If we found only 1 drone
            elif found_drone and len(received_messages) == 1:
                (nearest_position1, nearest_angle1) = nearest_drone_coordinate1
                diff_angle1 = normalize_angle(nearest_angle1 - self.measured_angle())
                diff_angle = diff_angle1

            # if you are far away, you get closer
            # heading < 0: at left
            # heading > 0: at right
            # rotation_velocity : -1:left, 1:right
            # we are trying to align : diff_angle -> 0
            command_comm[self.rotation_velocity] = sign(diff_angle)

            # Desired distance between drones
            desired_dist = 60

            d1x = nearest_position1[0] - self.measured_position()[0]
            d1y = nearest_position1[1] - self.measured_position()[1]
            distance1 = math.sqrt(d1x ** 2 + d1y ** 2)

            d1 = distance1 - desired_dist
            # We use a logistic function. -1 < intensity1(d1) < 1 and  intensity1(0) = 0
            # intensity1(d1) approaches 1 (resp. -1) as d1 approaches +inf (resp. -inf)
            intensity1 = 2 / (1 + math.exp(-d1 / (desired_dist * 0.5))) - 1

            direction1 = math.atan2(d1y, d1x)
            heading1 = normalize_angle(direction1 - self.measured_angle())

            # The drone will slide in the direction of heading
            longi1 = intensity1 * math.cos(heading1)
            lat1 = intensity1 * math.sin(heading1)

            # If we found only 1 drone
            if found_drone and len(received_messages) == 1:
                command_comm[self.longitudinal_force] = longi1
                command_comm[self.lateral_force] = lat1

            # If we found at least 2 drones
            elif found_drone and len(received_messages) >= 2:
                d2x = nearest_position2[0] - self.measured_position()[0]
                d2y = nearest_position2[1] - self.measured_position()[1]
                distance2 = math.sqrt(d2x ** 2 + d2y ** 2)

                d2 = distance2 - desired_dist
                intensity2 = 2 / (1 + math.exp(-d2 / (desired_dist * 0.5))) - 1

                direction2 = math.atan2(d2y, d2x)
                heading2 = normalize_angle(direction2 - self.measured_angle())

                longi2 = intensity2 * math.cos(heading2)
                lat2 = intensity2 * math.sin(heading2)

                command_comm[self.longitudinal_force] = 0.5 * (longi1 + longi2)
                command_comm[self.lateral_force] = 0.5 * (lat1 + lat2)

        return found_drone, command_comm