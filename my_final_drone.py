"Drone that goes along the walls. If it detects a body, it will grasp it"

import random
import numpy as np
import cv2


from math import sqrt, atan, atan2, cos, exp, pi
from typing import Optional
from enum import Enum, auto

from spg_overlay.drone_abstract import DroneAbstract
from spg_overlay.drone_sensors import DroneSemanticCones
from spg_overlay.utils import normalize_angle

from solutions.algorithme_astar import astar

from solutions.KHT import *


def rot(angle):
    return np.matrix([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])


class MyWallPathDrone(DroneAbstract):
    class Activity(Enum):
        """
        Possible states of the drone
        """
        # States for the Leader drones
        SEARCHING_RIGHT = auto()
        SEARCHING_LEFT = auto()
        FOUND_WOUNDED_FAR = auto()
        FOUND_WOUNDED_NEAR = auto()
        FOUND_WOUNDED_SCAN = auto()
        GOING_BACK = auto()
        BACKUP_RIGHT = auto()
        BACKUP_LEFT = auto()
        GOING_BACK_BACK = auto()
        REPOSITIONING = auto()

        # States for the Rescuing drone
        STARTING = auto()
        GO_GETEM = auto()
        BRING_TO_RESCUE = auto()
        STAND_BY = auto()
        GO_TO_SLEEP = auto()

    class Type(Enum):
        LEADER = auto()
        FOLLOWER = auto()
        RESCUE = auto()

    def __init__(self, last_right_state: bool = None, identifier: Optional[int] = None, **kwargs):
        super().__init__(identifier=identifier,
                         should_display_lidar=False,
                         **kwargs)

        if self.identifier == 0 or self.identifier == 4:
            self.type = self.Type.LEADER
        elif self.identifier == 8:
            self.type = self.Type.RESCUE
        else:
            self.type = self.Type.FOLLOWER

        # State of the drone from the Enum Activity class
        if self.type is self.Type.LEADER:
            self.state = self.Activity.SEARCHING_RIGHT if self.identifier % 2 == 0 else self.Activity.SEARCHING_LEFT
        else:
            self.state = self.Activity.STARTING

        # Constants for the drone
        self.last_right_state = last_right_state
        self.base_speed = 0.3
        self.base_rot_speed = 1
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
        self.update_rate = 5
        self.save_rate = 100

        self.map = np.zeros(self.size_area)
        self.explored_map = np.ones((
            self.size_area[1]//self.scale, self.size_area[0]//self.scale))
        self.HSpace = HoughSpace([], self.size_area, 200, 200)
        self.nstep = 0
        self.n_local_step = 0
        self.last_20_pos = [(i, i) for i in range(100)]

        self.lidar_angles = self.lidar().ray_angles
        self.nb_angles = len(self.lidar_angles)

        self.rescue_HSpace = HoughSpace([], self.size_area, 200, 200)
        self.found_rescue_center = False
        self.rescue_center_pos = [0, 0]
        self.next_pos_to_go = []
        self.chief_pos = np.array([[0.0], [0.0]])

    def define_message(self):
        """
        Comm for the other drones
        """
        dest = self.identifier + 1
        try:
            x, y = self.get_pos()
            return [self.identifier, dest, np.array([x, y]), self.state]
        except:
            pass

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

    def process_communication_sensor_rescue(self):
        try:
            x, y = self.get_pos()
            pos = np.array([x, y])
        except:
            print("Erreur")
            return False

        if self.communication:
            messages = self.communication.received_message
            for msg in messages:
                try:
                    msg = msg[1]
                    other_pos = msg[2]
                    if np.sqrt(np.dot(pos - other_pos, pos - other_pos)) < 200 and msg[0] != 2:
                        return False

                except Exception:
                    pass
        return True

    def process_communication_sensor_explore(self):
        """ return False if an nearby ally drone is rescuing someone"""
        try:
            x, y = self.get_pos()
            pos = np.array([x, y])
        except:
            print("Erreur")
            return False

        if self.communication:
            messages = self.communication.received_message
            for msg in messages:
                try:
                    msg = msg[1]
                    other_pos = msg[2]
                    other_state = msg[3]
                    if np.sqrt(np.dot(pos - other_pos, pos - other_pos)) < 200 and msg[0] != self.identifier and (other_state in [self.Activity.FOUND_WOUNDED_FAR, self.Activity.FOUND_WOUNDED_NEAR, self.Activity.GOING_BACK, self.Activity.FOUND_WOUNDED_SCAN, self.Activity.GO_GETEM, self.Activity.BRING_TO_RESCUE]):
                        return False
                except Exception as e:
                    print(msg)
                    print("Sah le couz du douze")
                    print(e)
                    pass
        return True

    def process_communication_sensor_follower(self):

        if self.communication:
            received_messages = self.communication.received_message
            for message in received_messages:
                try:
                    message = message[1]
                    pos = np.array([[message[2][0]], [message[2][1]]])
                    if message[1] == self.identifier and message[0] == self.identifier - 1:
                        self.next_pos_to_go.append(pos)
                    elif message[1] == 0:
                        self.chief_pos = pos
                except:
                    print("error")
                    pass

    def merge_maps(self, other_walls, other_exploration):
        for w in other_walls:
            if not len(self.HSpace.data_walls):
                self.HSpace.data_walls.append(w)

            else:
                new_data = []
                affected_walls = []

                for l in self.HSpace.data_walls:
                    res = l.compare_walls(w)
                    if res:
                        affected_walls.append(res)
                    else:
                        new_data.append(l)

                if (len(affected_walls)):
                    new_data.append(affected_walls[0])

                    if (len(affected_walls) > 1):
                        for a in affected_walls[1:]:
                            new_data[-1] = a.compare_walls(new_data[-1])

                else:
                    new_data.append(w)

                self.data_walls = new_data[:]

        self.explored_map = np.bitwise_and(self.explored_map, other_exploration)

    def filter_position(self):

        cur_pos = self.measured_position()
        if (len(self.time) >= 2):
            self.time.append(self.time[-1]+1)
            d_vit = self.measured_velocity()

            d_pos = 0.5*(np.array([[d_vit[0]], [d_vit[1]]])+self.l_vit[-1])
            k_pos = 0.006
            pos1 = self.l_pos[-1]+d_pos  # rot.dot(d_pos)
            pos2 = np.array([[cur_pos[0]], [cur_pos[1]]])

            mix = k_pos*pos2 + (1-k_pos) * \
                pos1 if not self.gps_is_disabled() else pos1
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

    def save_map(self):
        if (self.debug_id != self.identifier):
            return

        self.HSpace.point_transform()
        self.HSpace.compute_lines_length()
        try:
            map = self.draw_map(self.scale)
        except:
            pass

    def draw_map(self, scale):
        size = [self.size_area[1] // scale, self.size_area[0] // scale]
        map = np.zeros(size)

        for line in self.HSpace.data_walls:
            if line.orientation:    # Droite verticale
                for y in range(round(line.p2[1] // scale), min(round(line.p1[1] // scale) + 1, size[0])):
                    map[y, round(0.5 * (line.p1[0] + line.p2[0]))//scale] = 1

            else:
                for x in range(round(line.p1[0] // scale), min(round(line.p2[0] // scale) + 1, size[1])):
                    map[round(0.5 * (line.p1[1] + line.p2[1]))//scale, x] = 1

        return map

    def update_map(self, lidar_sensor):
        # drone_pos = np.array(self.true_position()).reshape(2, 1)
        # drone_angle = self.true_angle()

        drone_pos = self.l_pos[-1]
        drone_angle = self.measured_angle()

        semantic_data = self.semantic_cones().sensor_values
        semantic_angle = semantic_data[0].angle
        semantic_angle_id = 0

        x, y = self.get_pos()
        x = int(x//self.scale)
        y = int(y//self.scale)
        for i in range(-50//self.scale, 50//self.scale):
            for j in range(-50//self.scale, 50//self.scale):
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
                    if (semantic_data[semantic_angle_id].entity_type != DroneSemanticCones.TypeEntity.DRONE) and (semantic_data[semantic_angle_id].entity_type != DroneSemanticCones.TypeEntity.WOUNDED_PERSON):

                        if not self.nstep:
                            self.map[round(x_cor), round(y_cor)] = 1
                            new_points.append([x_cor, y_cor])
                        else:
                            self.accumulator_map[round(
                                y_cor // self.scale), round(x_cor // self.scale)] += 1
                            if self.accumulator_map[round(y_cor // self.scale), round(x_cor // self.scale)] == 4:
                                new_points.append(
                                    [(x_cor // self.scale) * self.scale, (y_cor // self.scale) * self.scale])

                    if (semantic_data[semantic_angle_id].entity_type == DroneSemanticCones.TypeEntity.RESCUE_CENTER) and (not self.nstep):
                        self.base_pos = (y, x)
                else:
                    if not self.nstep:
                        self.map[round(x_cor), round(y_cor)] = 2
                        new_points.append([x_cor, y_cor])
                    else:
                        self.accumulator_map[round(
                            y_cor // self.scale), round(x_cor // self.scale)] += 1
                        if self.accumulator_map[round(y_cor // self.scale), round(x_cor // self.scale)] == 4:
                            new_points.append(
                                [(x_cor // self.scale) * self.scale, (y_cor // self.scale) * self.scale])

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

    def is_stucked(self, last_20_pos):
        var = np.var(last_20_pos, axis=0)
        # return False
        return var[0] <= 0.1 and var[1] <= 0.1 and not self.state is self.Activity.FOUND_WOUNDED_SCAN

    def update_last_20_pos(self, last_20_pos, new_pos):
        last_20_pos.pop(0)
        last_20_pos.append(new_pos)

    def get_rescue_center(self):
        points = []

        drone_pos = self.l_pos[-1]
        drone_angle = self.measured_angle()

        for r in self.semantic_cones().sensor_values:
            if r.entity_type == DroneSemanticCones.TypeEntity.RESCUE_CENTER:
                obstacle_pos = r.distance * \
                    np.dot(rot(drone_angle + r.angle),
                           np.array([1, 0]).reshape(2, 1)) + drone_pos
                points.append(obstacle_pos)

        self.rescue_HSpace.add_points_to_process(points)

    def set_anchor_pos(self):
        lines = self.rescue_HSpace.compute_lines_length()
        tx, ty = self.get_pos()
        pos = np.array([tx, ty])

        if not len(lines):
            return False

        else:
            l = lines[0]
            x = int((l.p1[0] + l.p2[0]) // 2)
            y = int((l.p1[1] + l.p2[1]) // 2)

        ext_norm = (pos - self.rescue_center_pos)

        self.rescue_center_pos = np.array([x, y])
        self.stand_by_pos = self.rescue_center_pos + 50 * \
            ext_norm / np.sqrt(np.dot(ext_norm, ext_norm))

        return True

    def control_leader(self):
        self.filter_position()
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

        x, y = self.get_pos()
        x = min(int(x), self.size_area[0])//self.scale
        y = min(int(y), self.size_area[1])//self.scale
        self.update_last_20_pos(self.last_20_pos, (y, x))

        values = self.process_lidar_sensor(self.lidar())

        if self.state is self.Activity.SEARCHING_RIGHT:
            if not (self.nstep % self.update_rate):
                self.update_map(self.lidar())
            if not (self.nstep % 100):
                self.save_map()
                self.accumulator_map = np.zeros(self.accumulator_map.shape)
            wall_front = values[front_index] < distance_max
            wall_front_right = values[frontright_index] < distance_max
            wall_right = values[right_index] < distance_max
            wall_left = values[left_index] < distance_max
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
                    # command[self.longitudinal_force] = -self.base_speed/2
                elif y < x/cos(alpha) - self.epsilon:
                    command[self.rotation_velocity] = -self.base_rot_speed
                    # command[self.longitudinal_force] = -self.base_speed/2
                else:
                    command[self.longitudinal_force] = self.base_speed
            wounded_person_angle, wounded_person_distance = self.process_semantic_sensor_wounded(
                self.semantic_cones())
            if wounded_person_angle != 0 and self.process_communication_sensor_explore():
                self.state = self.Activity.FOUND_WOUNDED_FAR
        elif self.state is self.Activity.SEARCHING_LEFT:
            if not (self.nstep % self.update_rate):
                self.update_map(self.lidar())
            if not (self.nstep % 100):
                self.save_map()
                self.accumulator_map = np.zeros(self.accumulator_map.shape)
            wall_front = values[front_index] < distance_max
            wall_front_right = values[frontright_index] < distance_max
            wall_right = values[right_index] < distance_max
            wall_left = values[left_index] < distance_max
            if not wall_front and not wall_left:
                command[self.longitudinal_force] = self.base_speed
            elif wall_front:
                command[self.rotation_velocity] = self.base_rot_speed
            else:
                x = values[left_index]
                alpha = self.lidar_angles[frontleft_index]
                y = values[frontleft_index]
                if y > x/cos(alpha) + self.epsilon:
                    command[self.rotation_velocity] = -self.base_rot_speed
                    # command[self.longitudinal_force] = -self.base_speed/2
                elif y < x/cos(alpha) - self.epsilon:
                    command[self.rotation_velocity] = self.base_rot_speed
                    # command[self.longitudinal_force] = -self.base_speed/2
                else:
                    command[self.longitudinal_force] = self.base_speed

            wounded_person_angle, wounded_person_distance = self.process_semantic_sensor_wounded(
                self.semantic_cones())
            if wounded_person_angle != 0 and self.process_communication_sensor_explore():
                self.state = self.Activity.FOUND_WOUNDED_FAR

        elif self.state is self.Activity.FOUND_WOUNDED_FAR:
            if not (self.nstep % self.update_rate):
                self.update_map(self.lidar())
            if not (self.nstep % 100):
                self.save_map()
                self.accumulator_map = np.zeros(self.accumulator_map.shape)
            wounded_person_angle, wounded_person_distance = self.process_semantic_sensor_wounded(
                self.semantic_cones())
            if wounded_person_angle < 0:
                command[self.rotation_velocity] = -self.base_rot_speed
                command[self.longitudinal_force] = 0.1
            elif wounded_person_angle > 0:
                command[self.rotation_velocity] = self.base_rot_speed
                command[self.longitudinal_force] = 0.1
            if wounded_person_distance < 50:
                self.state = self.Activity.FOUND_WOUNDED_NEAR
            if wounded_person_distance < 20 and wounded_person_distance > 0:
                command[self.grasp] = 1
                self.state = self.Activity.FOUND_WOUNDED_SCAN
        elif self.state is self.Activity.FOUND_WOUNDED_NEAR:
            if not (self.nstep % self.update_rate):
                self.update_map(self.lidar())
            if not (self.nstep % 100):
                self.save_map()
                self.accumulator_map = np.zeros(self.accumulator_map.shape)
            wounded_person_angle, wounded_person_distance = self.process_semantic_sensor_wounded(
                self.semantic_cones())
            if wounded_person_angle < 0:
                command[self.rotation_velocity] = self.base_rot_speed
                command[self.longitudinal_force] = -0.1
            elif wounded_person_angle > 0:
                command[self.rotation_velocity] = -self.base_rot_speed
                command[self.longitudinal_force] = -0.1
            if wounded_person_distance < 20 and wounded_person_distance > 0:
                command[self.grasp] = 1
                self.state = self.Activity.FOUND_WOUNDED_SCAN
        elif self.state is self.Activity.FOUND_WOUNDED_SCAN:
            command[self.grasp] = 1
            command[self.rotation_velocity] = 0.2
            self.n_local_step += 1
            if not (self.nstep % self.update_rate):
                self.update_map(self.lidar())
            if not (self.nstep % 100):
                self.save_map()
                self.accumulator_map = np.zeros(self.accumulator_map.shape)
            if self.n_local_step > 80:
                command[self.rotation_velocity] = 0
                command[self.longitudinal_force] = 0.2
            if self.n_local_step > 100:
                command[self.longitudinal_force] = 0.2
                self.n_local_step = 0
                self.state = self.Activity.GOING_BACK

        elif self.state is self.Activity.GOING_BACK:
            command[self.grasp] = 1
            if self.path == []:
                # self.save_map()
                x, y = self.get_pos()
                map_local = self.draw_map(self.scale)
                x = min(int(x), self.size_area[0])//self.scale
                y = min(int(y), self.size_area[1])//self.scale
                try:
                    self.path = self.path_to_points(
                        astar(map_local+self.explored_map, (y, x), self.base_pos), map_local+self.explored_map)
                    print(self.path)
                    self.reverse_path = list(reversed(self.path))
                    self.reverse_path.pop(0)
                    self.reverse_path.append((y, x))
                except:
                    print(
                        "Astar a merdé, probablement à cause de l'incertitude des mesures")
                    # plt.imshow(map_local+self.explored_map)
                    # plt.show()
                    self.state = self.Activity.BACKUP_RIGHT if self.identifier % 2 == 0 else self.Activity.BACKUP_LEFT
            else:
                command[self.longitudinal_force] = 0.5
                destination = self.path[0]
                x, y = self.get_pos()
                x = min(int(x), self.size_area[0])//self.scale
                y = min(int(y), self.size_area[1])//self.scale
                x_diff = destination[1]-x
                y_diff = destination[0]-y
                alpha = 0
                if x_diff > 0:
                    alpha = atan(y_diff/x_diff) % (2*pi)
                elif x_diff < 0:
                    alpha = pi + atan(y_diff/x_diff)
                else:
                    alpha = 0
                alpha_diff = (alpha-self.measured_angle()) % (2*pi)
                if alpha_diff < pi:
                    command[self.rotation_velocity] += 0.7
                elif alpha_diff > 0:
                    command[self.rotation_velocity] -= 0.7
                if abs(x_diff)**2 + abs(y_diff)**2 <= 4:
                    self.path.pop(0)
                    if self.path == []:
                        self.state = self.Activity.GOING_BACK_BACK

        elif self.state is self.Activity.BACKUP_RIGHT:
            command[self.grasp] = 1
            wall_front = values[front_index] < distance_max
            wall_front_right = values[frontright_index] < distance_max
            wall_right = values[right_index] < distance_max
            wall_left = values[left_index] < distance_max
            if not wall_front and not wall_left:
                command[self.longitudinal_force] = self.base_speed
            elif wall_front:
                command[self.rotation_velocity] = self.base_rot_speed
            else:
                x = values[left_index]
                alpha = self.lidar_angles[frontleft_index]
                y = values[frontleft_index]
                if y > x/cos(alpha) + self.epsilon:
                    command[self.rotation_velocity] = -self.base_rot_speed
                    command[self.longitudinal_force] = -self.base_speed/2
                elif y < x/cos(alpha) - self.epsilon:
                    command[self.rotation_velocity] = self.base_rot_speed
                    command[self.longitudinal_force] = -self.base_speed/2
                else:
                    command[self.longitudinal_force] = self.base_speed
            if not self.nstep % 100:
                try:
                    self.path = []
                    x, y = self.get_pos()
                    map_local = self.draw_map(self.scale)
                    x = min(int(x), self.size_area[0])//self.scale
                    y = min(int(y), self.size_area[1])//self.scale
                    self.path = self.path_to_points(
                        astar(map_local+self.explored_map, (y, x), self.base_pos), map_local+self.explored_map)
                    print(self.path)
                    self.reverse_path = list(reversed(self.path))
                    self.reverse_path.pop(0)
                    self.reverse_path.append((y, x))
                    self.state = self.Activity.GOING_BACK
                except:
                    pass
        elif self.state is self.Activity.BACKUP_LEFT:
            command[self.grasp] = 1
            wall_front = values[front_index] < distance_max
            wall_front_right = values[frontright_index] < distance_max
            wall_right = values[right_index] < distance_max
            wall_left = values[left_index] < distance_max
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
                    command[self.longitudinal_force] = -self.base_speed/2
                elif y < x/cos(alpha) - self.epsilon:
                    command[self.rotation_velocity] = -self.base_rot_speed
                    command[self.longitudinal_force] = -self.base_speed/2
                else:
                    command[self.longitudinal_force] = self.base_speed
            if not self.nstep % 100:
                try:
                    self.path = []
                    x, y = self.get_pos()
                    map_local = self.draw_map(self.scale)
                    x = min(int(x), self.size_area[0])//self.scale
                    y = min(int(y), self.size_area[1])//self.scale
                    self.path = self.path_to_points(
                        astar(map_local+self.explored_map, (y, x), self.base_pos), map_local+self.explored_map)
                    print(self.path)
                    self.reverse_path = list(reversed(self.path))
                    self.reverse_path.pop(0)
                    self.reverse_path.append((y, x))
                    self.state = self.Activity.GOING_BACK
                except:
                    pass

        elif self.state is self.Activity.GOING_BACK_BACK:
            if self.reverse_path == []:
                self.state = self.Activity.SEARCHING_RIGHT if self.identifier % 2 == 0 else self.Activity.SEARCHING_LEFT
            else:
                command[self.longitudinal_force] = 0.5
                destination = self.reverse_path[0]
                x, y = self.get_pos()
                x = min(int(x), self.size_area[0])//self.scale
                y = min(int(y), self.size_area[1])//self.scale
                x_diff = destination[1]-x
                y_diff = destination[0]-y
                alpha = 0
                if x_diff > 0:
                    alpha = atan(y_diff/x_diff) % (2*pi)
                elif x_diff < 0:
                    alpha = pi + atan(y_diff/x_diff)
                else:
                    alpha = 0
                alpha_diff = (alpha-self.measured_angle()) % (2*pi)
                if alpha_diff < pi:
                    command[self.rotation_velocity] += 0.7
                elif alpha_diff > 0:
                    command[self.rotation_velocity] -= 0.7
                if abs(x_diff)**2 + abs(y_diff)**2 <= 4:
                    self.reverse_path.pop(0)
                    if self.reverse_path == []:
                        self.state = self.Activity.SEARCHING_RIGHT if self.identifier % 2 == 0 else self.Activity.SEARCHING_LEFT

        elif self.state is self.Activity.REPOSITIONING:
            self.path = []
            command[self.longitudinal_force] = -1
            command[self.lateral_force] = -1 if self.nstep % 2 == 0 else 1
            command[self.grasp] = 1
            self.last_20_pos = [(i, i) for i in range(100)]
            self.state = self.Activity.SEARCHING_RIGHT if self.identifier % 2 == 0 else self.Activity.SEARCHING_LEFT

        touch_value = self.process_touch_sensor()
        if touch_value == "left":
            command[self.lateral_force] = self.base_speed
        elif touch_value == "right":
            command[self.lateral_force] = -self.base_speed

        if self.is_stucked(self.last_20_pos):
            command[self.grasp] = 1
            self.state = self.Activity.REPOSITIONING

        self.nstep += 1

        return command

    def control_follower(self):
        self.filter_position()
        self.process_communication_sensor_follower()
        command = {self.longitudinal_force: 0.0,
                   self.lateral_force: 0.0,
                   self.rotation_velocity: 0.0,
                   self.grasp: 0}

        destination= self.l_pos[-1]
        distance_from_drone = 0.0
        ###############  Get the next pos to go depending on the leader instructions and some others shit #################################
        if len(self.next_pos_to_go) >= 1:
    
            destination = self.next_pos_to_go[0]

            distance_from_drone = np.linalg.norm(
                self.next_pos_to_go[-1]-self.l_pos[-1])

            look_shortcut = self.next_pos_to_go[:]
            look_shortcut.reverse()
            for n, i in enumerate(look_shortcut):
                d = np.linalg.norm(i - self.l_pos[-1])
                if d < 10:
                    self.next_pos_to_go = self.next_pos_to_go[len(look_shortcut) - n-2:]
                    break
        ############ Get the angle to compute ##############################
        x, y = self.l_pos[-1][0][0], self.l_pos[-1][1][0]
        x_diff = destination[0][0]-x
        y_diff = destination[1][0]-y
        
        alpha = atan2(y_diff, x_diff)
        alpha = normalize_angle(alpha)
        a2 = normalize_angle(self.measured_angle())
        alpha_diff = normalize_angle(alpha-a2)
        if alpha_diff < 0:
            command[self.rotation_velocity] -= 0.5
        elif alpha_diff > 0:
            command[self.rotation_velocity] += 0.5
            
        ######### Compute the cone sensor to see if there is drone around and correct his goal ###########################
        cones = self.semantic_cones().sensor_values
        l_supposed_pos = []
        is_alone = False
        print(np.linalg.norm(self.l_pos[-1] - np.array([[self.true_position()[0]],[self.true_position()[1]]])))

        destination = self.l_pos[-1]

        #if distance_from_drone > 100:
        #    self.next_pos_to_go.pop(0)

        
        command[self.longitudinal_force] = 0.4 - exp(-distance_from_drone/60)
            
        if is_alone and distance_from_drone >= 10:
            command[self.longitudinal_force] = max(
            0.4 - exp(-distance_from_drone/60), 0.1)
        return command

    def control_rescue(self):
        self.filter_position()
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

        x, y = self.get_pos()
        x = min(int(x), self.size_area[0])//self.scale
        y = min(int(y), self.size_area[1])//self.scale
        self.update_last_20_pos(self.last_20_pos, (y, x))

        values = self.process_lidar_sensor(self.lidar())

        if self.state == self.Activity.STARTING:
            if self.nstep < 30:
                self.get_rescue_center()

            elif not self.found_rescue_center:
                self.found_rescue_center = self.set_anchor_pos()

                if self.found_rescue_center:
                    self.state = self.Activity.GO_TO_SLEEP

        elif self.state == self.Activity.STAND_BY:
            if self.process_communication_sensor_rescue() and self.process_semantic_sensor_wounded(self.semantic_cones()):
                self.state = self.Activity.GO_GETEM

        elif self.state is self.Activity.GO_GETEM:
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

            rescue_center_angle, rescue_center_distance = alpha_diff, np.sqrt(
                np.dot(pos - self.rescue_center_pos, pos - self.rescue_center_pos))

            if rescue_center_angle < 0:
                command[self.rotation_velocity] = -self.base_rot_speed
                command[self.longitudinal_force] = 0.1
            elif rescue_center_angle > 0:
                command[self.rotation_velocity] = self.base_rot_speed
                command[self.longitudinal_force] = 0.1

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

            stand_by_angle, stand_by_distance = alpha_diff, np.sqrt(
                np.dot(pos - self.stand_by_pos, pos - self.stand_by_pos))

            if stand_by_angle < 0:
                command[self.rotation_velocity] = -self.base_rot_speed
                command[self.longitudinal_force] = 0.1
            elif stand_by_angle > 0:
                command[self.rotation_velocity] = self.base_rot_speed
                command[self.longitudinal_force] = 0.1

            if stand_by_distance < 20 and stand_by_distance > 0:
                command[self.grasp] = 1
                self.state = self.Activity.STAND_BY
            self.nstep += 1

        return command

    def control(self):
        if self.type is self.Type.LEADER:
            return self.control_leader()
        elif self.type is self.Type.FOLLOWER:
            return self.control_follower()
        elif self.type is self.Type.RESCUE:
            return self.control_rescue()
