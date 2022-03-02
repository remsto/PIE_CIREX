import random
from math import *
from copy import deepcopy
from re import L, S
from typing import Optional, Type
import matplotlib.pyplot as plt
import numpy as np
import time
from enum import Enum

from spg_overlay.drone_abstract import DroneAbstract
from spg_overlay.drone_sensors import DroneSemanticCones
from spg_overlay.misc_data import MiscData
from spg_overlay.utils import normalize_angle, sign


class MyDroneFilterPos(DroneAbstract):
    def __init__(self,
                 last_right_state: bool = None,
                 identifier: Optional[int] = None,
                 misc_data: Optional[MiscData] = None,
                 **kwargs):
        super().__init__(identifier=identifier,
                         misc_data=misc_data,
                         should_display_lidar=False,
                         **kwargs)
        self.counterStraight = 0
        self.angleStopTurning = 0
        self.isTurning = False
        self.map = np.zeros(self.size_area)
        self.l_pos = [np.array([[0.0], [0.0]])]
        self.chief_pos = np.array([[0.0], [0.0]])
        self.l_vit = []
        self.l_true_pos = []
        self.raw_data = []
        self.pos2 = []
        self.time = []
        self.stop = True
        ##################################################
        self.state = self.Activity.SEARCHING
        self.last_right_state = last_right_state
        self.base_speed = 0.2
        self.base_rot_speed = 0.1
        self.epsilon = 5
        self.scale = 10
        self.path = []
        self.reverse_path = []
        self.debug_id = self.identifier
        self.type = self.get_type()
        # A counter that allow the drone to turn for a specified amount of time
        self.turning_time = 0
        self.next_pos_to_go = []
        self.lidar_angles = self.lidar().ray_angles
        self.nb_angles = len(self.lidar_angles)

    def get_type(self):
        if self.identifier == 0:
            return self.Type.LEADER
        else:
            return self.Type.FOLLOWER

    class Activity(Enum):
        """
        Possible states of the drone
        """
        SEARCHING = 1
        FOUND = 2
        GOING_BACK = 3
        TURNING_RIGHT = 4
        TURNING_LEFT = 5
        STOP = 6

    class Type(Enum):
        LEADER = 1
        FOLLOWER = 2

    def filter_position(self):

        cur_pos = self.measured_position()
        if (len(self.l_pos) >= 2):
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
        return self.l_pos[-1]

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
            if data.entity_type == DroneSemanticCones.TypeEntity.DRONE:
                return data.angle, data.distance
        return 0.0, 0.0

    def process_lidar_sensor(self, the_lidar_sensor):
        """
        Returns the values of the lidar sensor
        """
        values = the_lidar_sensor.get_sensor_values()
        return values

    def control_follower(self):
        values = self.semantic_cones().sensor_values

        command = {self.longitudinal_force: 0.0,
                   self.lateral_force: 0.0,
                   self.rotation_velocity: 0.0,
                   self.grasp: 0}

        is_a_drone = False
        cones = self.semantic_cones().sensor_values
        for v in cones:
            if v.entity_type == DroneSemanticCones.TypeEntity.DRONE:
                is_a_drone = True
                break

        x, y = self.l_pos[-1][0][0], self.l_pos[-1][1][0]
        destination = self.l_pos[-1]
        distance_from_drone = 0
        if len(self.next_pos_to_go) >= 1:

            destination = self.next_pos_to_go[0]

            distance_from_drone = np.linalg.norm(
                self.next_pos_to_go[-1]-self.l_pos[-1])

            look_shortcut = self.next_pos_to_go[:]
            look_shortcut.reverse()
            for n, i in enumerate(look_shortcut):
                d = np.linalg.norm(i - self.l_pos[-1])
                if d < 7:
                    self.next_pos_to_go = self.next_pos_to_go[n-1:]
                    break

        x_diff = destination[0][0]-x
        y_diff = destination[1][0]-y
        # x = min(int(x), self.size_area[0])
        # y = min(int(y), self.size_area[1])

        alpha = atan2(y_diff, x_diff)
        alpha = normalize_angle(alpha)
        a2 = normalize_angle(self.measured_angle())
        alpha_diff = normalize_angle(alpha-a2)

        if alpha_diff < 0:
            command[self.rotation_velocity] -= 0.2
        elif alpha_diff > 0:
            command[self.rotation_velocity] += 0.2

        if distance_from_drone > 100:
            self.next_pos_to_go.pop(0)

        if not is_a_drone:
            distance_from_drone = 80

        d_chief = np.linalg.norm(self.chief_pos-self.l_pos[-1])
        if d_chief < 100:
            distance_from_drone = d_chief
        command[self.longitudinal_force] = max(
            0.4 - exp(-distance_from_drone/50), -0.2)
        return command

    def control_leader(self):

        command = {self.longitudinal_force: 0.0,
                   self.lateral_force: 0.0,
                   self.rotation_velocity: 0.0,
                   self.grasp: 0}
        distance_max = 20

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
               # command[self.rotation_velocity] = self.base_rot_speed
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

        if self.state is self.Activity.TURNING_LEFT:
            command = self.turn_90("left")

        if self.state is self.Activity.TURNING_RIGHT:
            command = self.turn_90("right")

        return command

    def control(self):
        self.filter_position()
        self.process_communication_sensor()
        if self.type == self.Type.FOLLOWER:
            return self.control_follower()
        else:
            return self.control_leader()

    def process_communication_sensor(self):

        if self.communication:
            received_messages = self.communication.received_message
            for message in received_messages:
                try:
                    message = message[1]
                    if self.type == self.Type.FOLLOWER:
                        if message[1] == self.identifier and message[0] == self.identifier - 1:
                            self.next_pos_to_go.append(message[2])
                        elif message[1] == 0:
                            self.chief_pos = message[2]
                except:
                    print("error")
                    pass

    def define_message(self):
        """
        Define the message, the drone will send to and receive from other surrounding drones.
        """
        dest = self.identifier + 1
        try:
            msg_data = (self.identifier, dest,
                        self.l_pos[-1])
        except:
            msg_data = (self.identifier, dest,
                        None)
        return msg_data


"""




    def control(self, elapsed_time):

        print(self.identifier)
        true_position = self.filter_position()
        if(self.identifier == 0):
            if (self.init):
                command = {self.longitudinal_force: 0.0,
                           self.lateral_force: 0.0,
                           self.rotation_velocity: 0.0}
            else:

        else:

            # if elapsed_time < 30:
            #    return command, False
        command_lidar, collision_lidar = self.process_lidar_sensor(
            self.lidar())
        found, command_comm = self.process_communication_sensor()


        alpha = 0.4
        alpha_rot = 0.75

        if collision_lidar:
            alpha_rot = 0.1

        # The final command  is a combinaison of 2 commands
        command[self.longitudinal_force] = alpha * command_comm[self.longitudinal_force] \
            + (1 - alpha) * command_lidar[self.longitudinal_force]
        command[self.lateral_force] = alpha * command_comm[self.lateral_force] \
            + (1 - alpha) * command_lidar[self.lateral_force]
        command[self.rotation_velocity] = alpha_rot * command_comm[self.rotation_velocity] \
            + (1 - alpha_rot) * command_lidar[self.rotation_velocity]

        return command

    # def get_position(self):"""
