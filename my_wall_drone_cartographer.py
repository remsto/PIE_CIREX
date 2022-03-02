"Drone that goes along the walls. If it detects a body, it will grasp it"

import random
import cv2
import numpy as np


from math import cos
from typing import Optional
from enum import Enum

from spg_overlay.drone_abstract import DroneAbstract
from spg_overlay.drone_sensors import DroneSemanticCones
from spg_overlay.utils import normalize_angle
from spg_overlay.misc_data import MiscData

from .KHT import *

def rot(angle):
        return np.matrix([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

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

    def __init__(self,
                last_right_state: bool = None,
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
        self.l_pos = []
        self.l_vit = []
        self.l_true_pos = []
        self.raw_data = []
        self.pos2 = []
        self.time = []

        self.scale = 5
        self.accumulator_map = np.zeros((round(self.size_area[1] // self.scale) + 1, round(self.size_area[0] // 5) + 1))
        self.visited_doors = []

        # State of the drone from the Enum Activity class
        self.state = self.Activity.SEARCHING

        # Constants for the drone
        self.last_right_state = last_right_state
        self.base_speed = 0.5
        self.base_rot_speed = 0.3
        self.epsilon = 5

        # A counter that allow the drone to turn for a specified amount of time
        self.turning_time = 0

        self.lidar_angles = self.lidar().ray_angles
        self.nb_angles = len(self.lidar_angles)

    def draw_map(self, scale):
        size = [self.size_area[1] // scale, self.size_area[0] // scale]
        map = np.zeros(size)

        for line in self.HSpace.data_walls:
            if line.orientation:    # Droite verticale
                for y in range(round(line.p2[1] // scale), round(line.p1[1] // scale) + 1):
                    map[y, round(0.5 * (line.p1[0] + line.p2[0]) // scale)] = 1

            else:
                for x in range(round(line.p1[0] // scale), round(line.p2[0] // scale) + 1):
                    map[round(0.5 * (line.p1[1] + line.p2[1]) // scale), x] = 1

        return map

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
        # return self.l_pos[-1]

    def update_map(self, lidar_sensor):
        # drone_pos = np.array(self.true_position()).reshape(2, 1)
        # drone_angle = self.true_angle()

        drone_pos = self.l_pos[-1]
        drone_angle = self.measured_angle()

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
                    if (semantic_data[semantic_angle_id].entity_type != DroneSemanticCones.TypeEntity.DRONE) and (semantic_data[semantic_angle_id].entity_type != DroneSemanticCones.TypeEntity.WOUNDED_PERSON):
                        
                        if not self.nstep:
                            self.map[round(x_cor), round(y_cor)] = 1
                            new_points.append([x_cor, y_cor])
                        else:
                            self.accumulator_map[round(y_cor // 5), round(x_cor // 5)] += 1
                            if self.accumulator_map[round(y_cor // 5), round(x_cor // 5)] == 4:
                                new_points.append([(x_cor // 5) * 5, (y_cor // 5) * 5])

                    if (semantic_data[semantic_angle_id].entity_type == DroneSemanticCones.TypeEntity.RESCUE_CENTER) and (not self.nstep):
                        self.base_pos = (round(y_cor // 5), round(x_cor // 5))
                else:
                    if not self.nstep:
                        self.map[round(x_cor), round(y_cor)] = 2
                        new_points.append([x_cor, y_cor])
                    else:
                        self.accumulator_map[round(y_cor // 5), round(x_cor // 5)] += 1
                        if self.accumulator_map[round(y_cor // 5), round(x_cor // 5)] == 4:
                                new_points.append([(x_cor // 5) * 5, (y_cor // 5) * 5])

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

    @mes_perf
    def save_map(self):
        if (self.debug_id != self.identifier):
            return

        # W, H = self.size_area
        # NOIR = "0 0 0\n"
        # BLANC = "255 255 255\n"
        # ROUGE = "255 0 0\n"

        # Color = [BLANC, NOIR, ROUGE]

        # with open("cartographer_drone_map.ppm", "w") as f:
        #     f.write("P3\n")
        #     f.write(f"{W} {H}\n255\n")
        #     for i in range(H):
        #         for j in range(W):
        #             f.write(Color[int(self.map[j, i])])

        # img =   cv2.imread("/home/antoine/PIE/swarm-rescue/src/swarm_rescue/cartographer_drone_map.ppm")
        # self.HSpace.background = img

        self.HSpace.point_transform()
        # self.HSpace.draw_90deg_lines_length()
        self.HSpace.compute_lines_length()
        map = self.draw_map(5)

        # plt.imshow(map)
        # plt.show()

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

    def process_semantic_sensor(self, the_semantic_sensor):
        """
        Returns the angle where a wounded person is, and 0 if not found
        """
        values = the_semantic_sensor.sensor_values

        for data in values:
            if data.entity_type == DroneSemanticCones.TypeEntity.WOUNDED_PERSON:
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

    def get_next_target(self):

        def middle(line):
            return np.array([(line.p1[0] + line.p2[0])/2, (line.p1[1] + line.p2[1])/2])

        pos = self.l_pos[-1].reshape((1, 2))

        doors = self.HSpace.data_doors
        door_index, door_pos = 0, middle(doors[0])
        min_dist = np.sqrt(np.power(door_pos - pos, 2))

        for i, door in enumerate(doors[1:]):
            if door.visited:
                continue

            p = middle(door)
            d = np.sqrt(np.power(p - pos, 2))

            if d < min_dist:
                door_index = i + 1
                door_pos = p
                min_dist = d

        return [door_index, door_pos]

    def correct_door_to_wall(self, door_pos):
        pass

    def control(self):

        command = {self.longitudinal_force: 0.0,
                   self.lateral_force: 0.0,
                   self.rotation_velocity: 0.0,
                   self.grasp: 0}
        distance_max = 20

        self.filter_position()

        if not (self.nstep % 2):
            self.update_map(self.lidar())
        if not (self.nstep % 50):
            self.save_map()
            self.accumulator_map = np.zeros(self.accumulator_map.shape)
        self.nstep += 1



        values = self.process_lidar_sensor(self.lidar())
        if self.state is self.Activity.SEARCHING:

            wall_front = values[44] < distance_max
            wall_front_right = values[70] < distance_max
            wall_right = values[66] < distance_max

            if not wall_front and not wall_right:
                command[self.longitudinal_force] = self.base_speed
            elif wall_front:
                command[self.rotation_velocity] = -self.base_rot_speed
            else:
                x = values[66]
                alpha = self.lidar_angles[55]
                y = values[55]
                if y > x/cos(alpha) + self.epsilon:
                    command[self.rotation_velocity] = self.base_rot_speed
                elif y < x/cos(alpha) - self.epsilon:
                    command[self.rotation_velocity] = -self.base_rot_speed
                else:
                    command[self.longitudinal_force] = self.base_speed

            # First prototype of Saving
            # wounded_person_angle, wounded_person_distance = self.process_semantic_sensor(
            #     self.semantic_cones())
            # if wounded_person_angle != 0:
            #     pass
            #     self.state = self.Activity.FOUND

        elif self.state is self.Activity.FOUND:
            wounded_person_angle, wounded_person_distance = self.process_semantic_sensor(
                self.semantic_cones())
            if wounded_person_angle < 0:
                command[self.rotation_velocity] = -self.base_rot_speed
                command[self.longitudinal_force] = self.base_speed
            elif wounded_person_angle > 0:
                command[self.rotation_velocity] = self.base_rot_speed
                command[self.longitudinal_force] = self.base_speed
            if wounded_person_distance < 20:
                command[self.grasp] = 1
                self.state = self.Activity.GOING_BACK

            # End of first prototype of saving
        elif self.state is self.Activity.GOING_BACK:

            command[self.grasp] = 1

            wall_front = values[44] < distance_max
            wall_front_right = values[70] < distance_max
            wall_left = values[0] < distance_max

            if not wall_front and not wall_left:
                command[self.longitudinal_force] = self.base_speed
            # elif wall_front:
               #command[self.rotation_velocity] = self.base_rot_speed
            else:
                x = values[0]
                alpha = self.lidar_angles[23]
                y = values[23]
                if y > x/cos(alpha) + self.epsilon:
                    command[self.rotation_velocity] = -self.base_rot_speed
                elif y < x/cos(alpha) - self.epsilon:
                    command[self.rotation_velocity] = self.base_rot_speed
                else:
                    command[self.longitudinal_force] = self.base_speed

        # touch_value = self.process_touch_sensor()
        # if touch_value == "left":
        #     command[self.lateral_force] = self.base_speed
        # elif touch_value == "right":
        #     command[self.lateral_force] = -self.base_speed

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
