#%%
from curses.ascii import NL
import math
from typing import Iterable, List
from cv2 import COLOR_BGR2GRAY, CV_8UC1, IMREAD_ANYCOLOR, IMREAD_GRAYSCALE, WINDOW_AUTOSIZE
import numpy as np
from scipy.linalg import eig
import scipy.ndimage as sci
import scipy.signal as sig

import matplotlib.image as im, matplotlib.pyplot as plt

#%%
class Pixel():

    def __init__(self, index_x, index_y, x, y):
        self.ix = index_x
        self.iy = index_y
        self.x = x
        self.y = y
class Cluster():

    def __init__(self, pixels):
        self.pixels = pixels
def find_chains(img, size):
    pass
def canny(img, low, high, sgm = 3): # L'image fournie sera toujours en noir et blanc
    img = sci.gaussian_filter(img, sigma=sgm)
    img = sci.sobel(img)
    return img

# img = im.imread("/home/antoine/PIE/swarm-rescue/cartographer_drone_map.png")
# print(img.shape)
# img = canny(img, 80, 200)
# plt.imshow(img)
# plt.show()

def build_hough_space_fom_image(img, shape = (100, 300), val = 1):
    hough_space = np.zeros((round(np.sqrt(img.shape[0] ** 2 + img.shape[1] ** 2)), round(np.sqrt(img.shape[0] ** 2 + img.shape[1] ** 2))))
    for i, row in enumerate(img):
        for j, pixel in enumerate(row):   
            if pixel[0] != val : continue
            hough_space = add_to_hough_space_polar((i,j), hough_space)
    return hough_space
def add_to_hough_space_polar(p, feature_space):
    theta = np.linspace(0, np.pi, feature_space.shape[0])
    d = p[0] * np.sin(theta) + p[1] * np.cos(theta)
    for i in range(feature_space.shape[0]):
        feature_space[round(d[i]), i] += 1
    return feature_space
# img = build_hough_space_fom_image(img, img.shape, 0)
def find_peaks(img, threshold=50):
    neighbourhood = sci.morphology.generate_binary_structure(2, 20)

    local_max = sci.maximum_filter(img, footprint=neighbourhood)
    local_min = sci.minimum_filter(img, footprint=neighbourhood)
    diff = ((local_max - local_min) >= threshold)

    return np.multiply(img, diff)

# img_2 = find_peaks(img)
# fig = plt.figure()
# ax = fig.add_subplot(1, 2, 1)
# imgplot = plt.imshow(img)
# ax.set_title('Before')
# ax = fig.add_subplot(1, 2, 2)
# imgplot = plt.imshow(img_2)
# ax.set_title('After')
# plt.show()
#%%
class Accumulator():

    def __init__(self, theta_bins, r_bins, max_theta, max_r):
        self.d_theta = max_theta / theta_bins
        self.d_r = max_r / r_bins

        self.max_theta = max_theta
        self.max_r = max_r

        self.r_bins = r_bins
        self.t_bins = theta_bins 

        ### TODO : ajouter un anneau autour avec les symétries pour la convolution

        self.shape = (r_bins, theta_bins)

        self.accumulator = np.zeros((2 * self.r_bins + 2, self.t_bins + 2))   # r peut être négatif

    def reset_accumulator(self):
        self.accumulator = np.zeros((2 * self.r_bins, self.t_bins))

    def __getitem__(self, key):
        if (len(key) != 2): raise IndexError

        r, theta = key
        r += self.r_bins

        if (r >= (2 * self.r_bins) or r < 0 or theta >= self.t_bins or theta < 0): raise IndexError

        return self.accumulator[int(r) + 1, int(theta) + 1]

    def __setitem__(self, key, value):
        if (len(key) != 2): raise IndexError

        r, theta = key
        nR = int(r)
        r += self.r_bins

        if (r >= (2 * self.r_bins) or r < 0 or theta >= self.t_bins or theta < 0): raise IndexError

        self.accumulator[int(r) + 1, int(theta) + 1] = value

        if (theta == 0):
            self.accumulator[-nR + self.r_bins, self.t_bins + 1] = value
        if (theta == (self.t_bins - 1)):
            self.accumulator[-nR + self.r_bins, 0] = value

    def get_maxima_90degrees(self, n = math.inf, threshold = 10, theta_res=0.07, t_width=1):
        array_0deg = np.concatenate((np.flip(self.accumulator[1:-1, (-t_width-1):-1]), self.accumulator[1:-1, 1:(t_width+2)]), axis = 1)
        array_90deg = self.accumulator[1:-1, (int(math.ceil(self.t_bins/2))-t_width+1):(int(math.ceil(self.t_bins/2))+t_width+1)]  # self.t_bins doit être pair pour que la répartition soit symétrique

        votes_0deg = [0]
        votes_90deg = [np.sum(array_90deg[:1, :])]
            
        for k in range(1, array_0deg.shape[0]-1):
            block_0deg = array_0deg[k:(k+2*t_width + 1), :]
            conv = sci.convolve(block_0deg, [[1, 2, 1], [2, 4, 2], [1, 2, 1]])[1, 1]
            votes_0deg.append(conv)

            block_90deg = array_90deg[k:(k+2*t_width), :]
            votes_90deg.append(np.sum(block_90deg))

        votes_0deg.append(0)
        votes_90deg.append(np.sum(array_90deg[-2:, :]))

        # fig = plt.figure()
        # ax = fig.add_subplot(1, 2, 1)
        # plt.plot(votes_0deg)
        # ax.set_title('0deg')
        # ax = fig.add_subplot(1, 2, 2)
        # plt.plot(votes_90deg)
        # ax.set_title('90deg')
        # plt.show()

        horiz_lines, _ = sig.find_peaks(votes_0deg, distance = 4, height=40)
        vert_lines, _ = sig.find_peaks(votes_90deg, distance = 4, height=10)

        lines = []

        for v in vert_lines:
            r = (v - self.r_bins) * self.d_r
            t = np.pi / 2
            lines.append((r, t))

        for h in horiz_lines:
            r = (h - self.r_bins + 1) * self.d_r
            t = (r < 0) * np.pi
            lines.append((r, t))

        return lines

    def get_maxima(self, n = math.inf, threshold = 2000):
        non_empty_bins = []
        votes_bins = []
        R, T = self.shape

        for r in range(2*self.shape[0]):
            for t in range(self.shape[1]):
                if self.accumulator[r+1, t+1] >= threshold:
                    non_empty_bins.append((r - self.r_bins, t, self.accumulator[r+1, t+1]))

        for b in non_empty_bins:
            r, t = b[0:2]
            nR = r + 1 + self.r_bins
            nT = t + 1

            bin_env = [[self.accumulator[nR-1, nT-1], self.accumulator[nR-1, nT], self.accumulator[nR-1, nT+1]], 
                       [self.accumulator[nR, nT-1],  self.accumulator[nR, nT], self.accumulator[nR, nT+1]],
                       [self.accumulator[nR+1, nT-1], self.accumulator[nR+1, nT], self.accumulator[nR+1, nT+1]]]

            conv = sci.convolve(bin_env, [[1, 2, 1], [2, 4, 2], [1, 2, 1]])[1, 1]

            votes_bins.append((r, t, conv))

        votes_bins.sort(key = lambda x: x[2], reverse=True)

        visit_map = np.zeros(self.accumulator.shape, bool)

        lines = []

        for b in votes_bins:
            r, t = b[0:2]
            nR = r + self.r_bins + 1
            nT = t + 1
            
            if (visit_map[nR+1, nT+1] or visit_map[nR, nT+1] or visit_map[nR+1, nT+1] or visit_map[nR+1, nT] or visit_map[nR, nT] or visit_map[nR+1, nT] or visit_map[nR+1, nT-1] or visit_map[nR, nT-1] or visit_map[nR+1, nT-1]):
                continue

            if (t == 0):
                if (visit_map[-nR + self.r_bins, self.t_bins + 1] or visit_map[-nR + self.r_bins + 1, self.t_bins + 1] or visit_map[-nR + self.r_bins - 1, self.t_bins + 1]):
                    continue                

            elif (t == (self.t_bins - 1)):
                if (visit_map[-nR + self.r_bins, 0] or visit_map[-nR + self.r_bins + 1, 0] or visit_map[-nR + self.r_bins - 1, 0]):
                    continue

            lines.append((r * self.d_r, t * self.d_theta))

            visit_map[nR+1, t+1] = True
            visit_map[nR, t+1] = True
            visit_map[nR+1, t+1] = True
            visit_map[nR+1, t] = True
            visit_map[nR, t] = True
            visit_map[nR+1, t] = True
            visit_map[nR+1, t-1] = True
            visit_map[nR, t-1] = True
            visit_map[nR+1, t-1] = True

            if (t == 0):
                visit_map[-nR + self.r_bins, self.t_bins + 1] = True
                visit_map[-nR + self.r_bins + 1, self.t_bins + 1] = True
                visit_map[-nR + self.r_bins - 1, self.t_bins + 1] = True

            elif (t == (self.t_bins - 1)):
                visit_map[-nR + self.r_bins, 0]
                visit_map[-nR + self.r_bins + 1, 0]
                visit_map[-nR + self.r_bins - 1, 0]

            if len(lines) == n:
                break

        return lines

class Line():

    def __init__(self, orientation, p1, p2, label="", visited=False):
        self.orientation = orientation # Horizontal : False, Vertical : True.
        self.p1 = p1
        self.p2 = p2
        self.label = label
        self.visited = visited

    def __str__(self):
        orientations = ["Horizontal", "Vertical"]
        return f"<{orientations[int(self.orientation)]} line between {self.p1} and {self.p2}>"

    def __repr__(self):
        return self.__str__()

    def compare_walls(self, other):
        if self.label != other.label:
            return False

        if self.orientation != other.orientation:
            return False        

        if self.orientation:
            if np.mean([abs(self.p1[0] - other.p1[0]), abs(self.p1[0] - other.p2[0]), abs(self.p2[0] - other.p1[0]), abs(self.p2[0] - other.p2[0])]) >= 10:
                return False

            if not ((other.p1[1] < (self.p2[1] - 20)) or (self.p1[1] < (other.p2[1] - 20))):  # Droites avec union non nul
                y_max = max(self.p1[1], other.p1[1])
                y_min = min(self.p2[1], other.p2[1])
                x = np.mean([self.p1[0], self.p2[0], other.p1[0], other.p2[0]])
                nline = Line(self.orientation, [x, y_max], [x, y_min], self.label)

                return nline

            else:
                return False

        else:
            if np.mean([abs(self.p1[1] - other.p1[1]), abs(self.p1[1] - other.p2[1]), abs(self.p2[1] - other.p1[1]), abs(self.p2[1] - other.p2[1])]) >= 10:
                return False

            if not ((other.p1[0] > (self.p2[0] + 20)) or (self.p1[0] > (other.p2[0] + 20))):  # Droites avec union non nul
                x_min = min(self.p1[0], other.p1[0])
                x_max = max(self.p2[0], other.p2[0])
                y = np.mean([self.p1[1], self.p2[1], other.p1[1], other.p2[1]])
                nline = Line(self.orientation, [x_min, y], [x_max, y], self.label)

                return nline

            else:
                return False

    def compare_doors(self, other):
        if (self.label != "wall") and (other.label != "wall"):
            return False

        if self.orientation != other.orientation:
            return False        

        if self.orientation:
            if np.mean([abs(self.p1[0] - other.p1[0]), abs(self.p1[0] - other.p2[0]), abs(self.p2[0] - other.p1[0]), abs(self.p2[0] - other.p2[0])]) >= 10:
                return False

            if not ((other.p1[1] < self.p2[1]) or (self.p1[1] < other.p2[1])):  # Droites avec union non nul
                y_min = max(self.p1[1], other.p1[1])
                y_max = min(self.p2[1], other.p2[1])
                x = np.mean([self.p1[0], self.p2[0], other.p1[0], other.p2[0]])
                nline = Line(self.orientation, [x, y_max], [x, y_min], self.label, self.visited or other.visited)

                return nline

            else:
                return False

        else:
            if np.mean([abs(self.p1[1] - other.p1[1]), abs(self.p1[1] - other.p2[1]), abs(self.p2[1] - other.p1[1]), abs(self.p2[1] - other.p2[1])]) >= 10:
                return False

            if not ((other.p1[0] > self.p2[0]) or (self.p1[0] > other.p2[0])):  # Droites avec union non nul
                x_min = max(self.p1[0], other.p1[0])
                x_max = min(self.p2[0], other.p2[0])
                y = np.mean([self.p1[1], self.p2[1], other.p1[1], other.p2[1]])
                nline = Line(self.orientation, [x_min, y], [x_max, y], self.label, self.visited or other.visited)

                return nline

            else:
                return False

    def compare_door_wall(self, other): # Self = door, other = wall
        if (self.label == "wall") and (other.label == "door"):
            return other.compare_door_wall(self)

        if (self.label != "door") and (other.label != "wall"):
            return False

        if self.orientation != other.orientation:
            return False        

        if self.orientation:
            if np.mean([abs(self.p1[0] - other.p1[0]), abs(self.p1[0] - other.p2[0]), abs(self.p2[0] - other.p1[0]), abs(self.p2[0] - other.p2[0])]) > 5:
                return False

            return not ((other.p1[1] < (self.p2[1] + 10)) or (self.p1[1] < (other.p2[1] + 10)))  # Droites avec union non nul

        else:
            if np.mean([abs(self.p1[1] - other.p1[1]), abs(self.p1[1] - other.p2[1]), abs(self.p2[1] - other.p1[1]), abs(self.p2[1] - other.p2[1])]) > 5:
                return False

            return not ((other.p1[0] > (self.p2[0] - 10)) or (self.p1[0] > (other.p2[0] - 10)))  # Droites avec union non nul




class HoughSpace():

    def __init__(self, points, img_space_shape, r_bins, theta_bins, img = None):
        self.processed_points = []
        self.not_processed_points = points
        self.processed_points_cache = []

        self.img_space_shape = img_space_shape

        self.max_r = math.sqrt(img_space_shape[0] ** 2 + img_space_shape[1] ** 2)
        self.max_theta = np.pi

        # actual_r_bins = 2 * r_bins

        self.d_theta = self.max_theta / theta_bins
        self.d_r = self.max_r / r_bins

        self.accumulator = Accumulator(theta_bins, r_bins, self.max_theta, self.max_r)
        self.background = img

        self.data_walls = []
        self.data_doors = []

    def point_transform(self):
        for p in self.not_processed_points:
            theta = np.arange(0, self.accumulator.shape[1])
            d = p[0] * np.sin(self.d_theta * theta) + p[1] * np.cos(self.d_theta * theta)
            for r, angle in zip(d, theta):
                self.accumulator[r // self.d_r, angle] += 1

        self.processed_points.extend(self.not_processed_points[:])
        self.processed_points_cache = self.not_processed_points[:]
        self.not_processed_points.clear()

    def add_points_to_process(self, points):
        self.not_processed_points.extend(points)

    def reset_processed_points(self):
        self.not_processed_points.extend(self.processed_points)
        self.processed_points.clear()
        self.processed_points_cache.clear()
        self.accumulator.reset_accumulator()

    def get_lines(self, n=math.inf, threshold=2000):
        lines = self.accumulator.get_maxima(n, threshold)
        return lines

    def get_90deg_lines(self, threshold=10):
        lines = self.accumulator.get_maxima_90degrees(threshold=threshold)
        return lines

    def plot(self):
        plt.imshow(self.accumulator.accumulator[1:-1, 1:-1], extent=[0, 180, -self.accumulator.r_bins, self.accumulator.r_bins - 1])
        plt.colorbar()
        plt.show()
    
    def compute_lines_length(self):
        lines = self.accumulator.get_maxima_90degrees()
        nLines = []
        not_found_yet = []

        for rho, theta in lines:
            if (abs((np.pi / 2) - theta) >= (np.pi / 4)): # Gère le cas des droites horizontales
                x = np.linspace(0, self.img_space_shape[0] - 1, int(self.img_space_shape[0] // 10))
                y = (rho / np.cos(theta)) - np.tan(theta) * x
                tag = False  # Droite verticale : True, droite horizontale : False.

            else:   # Droites verticales
                y = np.linspace(self.img_space_shape[1] - 1, 0, int(self.img_space_shape[1] // 10))
                x = (rho / np.sin(theta)) - y / np.tan(theta)
                tag = True # Droite verticale : True, droite horizontale : False.

            gap = True
            lline = 0

            for xx, yy in zip(x, y):
                found = False
                if ((0 <= yy) and (yy < self.img_space_shape[1])) and ((0 <= xx) and (xx < self.img_space_shape[0])):
                    for p in self.processed_points_cache:
                        if (np.sqrt((p[0] - xx)**2 + (p[1] - yy)**2) <= 10):
                            if not found:
                                found = True
                                if gap:
                                    nLines.append([tag, p, p]) 
                                    lline = 0
                                    gap = False
                                else:
                                    nLines[-1][-1] = p
                            
                        else:
                            not_found_yet.append(p)
                    
                    # self.processed_points_cache = not_found_yet[:]
                    not_found_yet.clear()

                if not found:
                    lline += 1

                gap = gap or (lline > 2)

        actual_lines = []
        for tag, p1, p2 in nLines:
            if (np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2) >= 20):
                if tag:
                    if p1[1] > p2[1]:
                        actual_lines.append(Line(tag, p1, p2, "wall"))
                    else:
                        actual_lines.append(Line(tag, p2, p1, "wall"))

                else:
                    if p1[0] < p2[0]:
                        actual_lines.append(Line(tag, p1, p2, "wall"))
                    else:
                        actual_lines.append(Line(tag, p2, p1, "wall"))

                # Fusion dynamique des murs détectés

                if not len(self.data_walls):
                    self.data_walls.append(actual_lines[-1])

                else:
                    new_data = []
                    affected_walls = []

                    for index, l in enumerate(self.data_walls):
                        res = l.compare_walls(actual_lines[-1])
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
                        new_data.append(actual_lines[-1])

                    self.data_walls = new_data[:]

        HALLWAY_MAX_WIDTH = 120
        HALLWAY_MIN_WIDTH = 30
        
        found_doors = []
        lLen = 0

        self.data_doors.clear()

        for i, l1 in enumerate(self.data_walls[:-1]):
            t1, p11, p12 = l1.orientation, l1.p1, l1.p2
            for j, l2 in enumerate(self.data_walls[(i+1):]):
                t2, p21, p22 = l2.orientation, l2.p1, l2.p2
                if t1 == t2:  # Même orientation =    ||, :, =, --
                    if t1 and t2: #   || ou :
                        if (abs(p12[0] - p21[0]) < 10) or (abs(p11[0] - p22[0]) < 10): # Murs verticalement empilés
                            if (abs(p12[1] - p21[1]) > HALLWAY_MIN_WIDTH) and (abs(p12[1] - p21[1]) < HALLWAY_MAX_WIDTH):
                                found_doors.append(Line(t1, p12, p21, "door"))
                            elif (abs(p11[1] - p22[1]) > HALLWAY_MIN_WIDTH) and (abs(p11[1] - p22[1]) < HALLWAY_MAX_WIDTH):
                                found_doors.append(Line(t1, p22, p11, "door"))

                        elif not ((p21[1] > p12[1]) or (p11[1] > p22[1])):  #   Les murs forment un couloir
                            if (abs(p11[0] - p21[0]) > HALLWAY_MIN_WIDTH) and (abs(p11[0] - p21[0]) < HALLWAY_MAX_WIDTH): # Largeur du couloir OK
                                y_min, y_max = max(p11[1], p21[1]), min(p12[1], p21[1])
                                if (y_max - y_min) > HALLWAY_MIN_WIDTH: #  Couloir assez long
                                    found_doors.append(Line(t1, [min(p11[0], p21[0]), y_min], [max(p11[0], p21[0]), y_min], "hallway"))
                                    found_doors.append(Line(t1, [min(p11[0], p21[0]), y_max], [max(p11[0], p21[0]), y_max], "hallway"))

                    elif not (t1 or t2): #  = ou --
                        if (abs(p11[1] - p22[1]) < 10) or (abs(p12[1] - p21[1]) < 10): # Murs horizontalement empilés
                            if (abs(p12[0] - p21[0]) > HALLWAY_MIN_WIDTH) and (abs(p12[0] - p21[0]) < HALLWAY_MAX_WIDTH):
                                found_doors.append(Line(t1, p12, p21, "door"))
                            elif (abs(p11[0] - p22[0]) > HALLWAY_MIN_WIDTH) and (abs(p11[0] - p22[0]) < HALLWAY_MAX_WIDTH):
                                found_doors.append(Line(t1, p22, p11, "door"))

                        elif not ((p21[0] > p12[0]) or (p11[0] > p22[0])):  #   Les murs forment un couloir
                            if (abs(p11[1] - p21[1]) > HALLWAY_MIN_WIDTH) and (abs(p11[1] - p21[1]) < HALLWAY_MAX_WIDTH): # Hauteur du couloir OK
                                x_min, x_max = max(p11[0], p21[0]), min(p12[0], p21[0])
                                if (x_max - x_min) > HALLWAY_MIN_WIDTH: #  Couloir assez long
                                    found_doors.append(Line(t1, [x_min, min(p11[1], p21[1])], [x_min, max(p11[1], p21[1])], "hallway"))
                                    found_doors.append(Line(t1, [x_max, min(p11[1], p21[1])], [x_max, max(p11[1], p21[1])], "hallway"))
                
                else: # Orientations opposées : - |
                    if t1: # l1 verticale
                        if (p21[1] < (p11[1] + 10)) and (p21[1] > (p12[1] - 10)): # Mur horizontal "coupe" le mur vertical, marge de 10 pixels
                            if (abs(p11[0] - p21[0]) > HALLWAY_MIN_WIDTH) and (abs(p11[0] - p21[0]) < HALLWAY_MAX_WIDTH):
                                found_doors.append(Line(t2, [p11[0], p21[1]], p21, "door"))
                            elif (abs(p11[0] - p22[0]) > HALLWAY_MIN_WIDTH) and (abs(p11[0] - p22[0]) < HALLWAY_MAX_WIDTH):
                                found_doors.append(Line(t2, p22, [p11[0], p22[1]], "door"))
                        
                        elif (p12[0] > (p21[0] - 10)) and (p12[0] < (p21[0] + 10)): # Mur vertical "coupe" le mur horizontal
                            if (abs(p21[1] - p12[1]) > HALLWAY_MIN_WIDTH) and (abs(p21[1] - p12[1]) < HALLWAY_MAX_WIDTH):
                                found_doors.append(Line(t1, p12, [p12[0], p21[1]], "door"))
                            elif (abs(p21[1] - p11[1]) > HALLWAY_MIN_WIDTH) and (abs(p21[1] - p11[1]) < HALLWAY_MAX_WIDTH):
                                found_doors.append(Line(t1, [p11[0], p21[1]], p11, "door"))
                    
                    else: # l2 verticale
                        if (p11[1] < (p21[1] + 10)) and (p11[1] > (p22[1] - 10)): # Mur horizontal "coupe" le mur vertical
                            if (abs(p21[0] - p11[0]) > HALLWAY_MIN_WIDTH) and (abs(p21[0] - p11[0]) < HALLWAY_MAX_WIDTH):
                                found_doors.append(Line(t1, [p21[0], p11[1]], p11, "door"))
                            elif (abs(p21[0] - p12[0]) > HALLWAY_MIN_WIDTH) and (abs(p21[0] - p12[0]) < HALLWAY_MAX_WIDTH):
                                found_doors.append(Line(t1, p12, [p21[0], p12[1]], "door"))
                        
                        elif (p22[0] > (p11[0] - 10)) and (p22[0] < (p12[0] + 10)): # Mur vertical "coupe" le mur horizontal
                            if (abs(p11[1] - p22[1]) > HALLWAY_MIN_WIDTH) and (abs(p11[1] - p22[1]) < HALLWAY_MAX_WIDTH):
                                found_doors.append(Line(t2, [p22[0], p11[1]], p22, "door"))
                            elif (abs(p11[1] - p21[1]) > HALLWAY_MIN_WIDTH) and (abs(p11[1] - p21[0]) < HALLWAY_MAX_WIDTH):
                                found_doors.append(Line(t2, p21, [p21[0], p11[1]], "door"))

                if (len(found_doors) > lLen):
                    lLen = len(found_doors)

                    door_in_wall = False

                    for wall in self.data_walls:
                        if found_doors[-1].compare_door_wall(wall):
                            door_in_wall = True
                            break

                    if not door_in_wall:
                        if not len(self.data_doors):
                                self.data_doors.append(found_doors[-1])

                        else:
                            new_data = []
                            affected_doors = []

                            for index, l in enumerate(self.data_doors):
                                res = l.compare_doors(found_doors[-1])
                                if res:
                                    affected_doors.append(res)
                                else:
                                    new_data.append(l)

                            if (len(affected_doors)):
                                new_data.append(affected_doors[0])

                                if (len(affected_doors) > 1):
                                    for a in affected_doors[1:]:
                                        new_data[-1] = a.compare_doors(new_data[-1])

                            else:
                                new_data.append(found_doors[-1])

                            self.data_doors = new_data[:]

        # new_doors = []

        # if len(found_doors) < 2:
        #     new_doors = found_doors[:]

        # for index, l in enumerate(found_doors[:-1]):
        #     i = 0
        #     for d in found_doors[index+1:]:
        #         res = d.compare_doors(l)
        #         i += 1
        #         if res:
        #             break

        #     if i == (len(found_doors) - index - 1):
        #         new_doors.append(l)

        # # self.data_doors.extend(new_doors)
        # self.data_doors = new_doors[:]

        ret = []
        ret.extend(self.data_walls)
        ret.extend(self.data_doors)

        return ret

    def compute_lines(self, lines, n=math.inf, threshold=2000):
        for rho, theta in lines:
            if (abs((np.pi / 2) - theta) >= (np.pi / 4)): # Gère le cas des droites verticales
                x = np.linspace(0, self.img_space_shape[0] - 1, self.img_space_shape[0])
                y = (rho / np.cos(theta)) - np.tan(theta) * x

                for xx, yy in zip(x, y):
                    if ((0 <= yy) and (yy < self.background.shape[0])):
                        self.background[int(yy), int(xx)] = (255, 255, 0)

            else:   # Droites horizontales
                y = np.linspace(0, self.img_space_shape[1] - 1, self.img_space_shape[1])
                x = (rho / np.sin(theta)) - y / np.tan(theta)
                
                for xx, yy in zip(x, y):
                    if ((0 <= xx) and (xx < self.background.shape[1])):
                        self.background[int(yy), int(xx)] = (255, 0, 255)               

    def draw_lines(self, n=math.inf, threshold=2000):
        self.compute_lines(self.get_lines(n, threshold), n, threshold)
        imgplot = plt.imshow(self.background)
        plt.colorbar()
        plt.title('Hough transform results')
        plt.show()

    def draw_90deg_lines(self, n=math.inf, threshold=2000):
        self.compute_lines(self.get_90deg_lines(threshold), n, threshold)
        imgplot = plt.imshow(self.background)
        plt.colorbar()
        plt.title('Hough transform results - for 0 and 90 degrees lines')
        plt.show()

    def draw_90deg_lines_length(self, n=math.inf, threshold=2000):
        res = self.compute_lines_length()

        color_config = {"wall":"yellow", "door":"dodgerblue", "hallway":"springgreen"}

        for index in range(len(res)):
            label, p1, p2 = res[index].label, res[index].p1, res[index].p2 
            plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color_config[label], linewidth=3)

        imgplot = plt.imshow(self.background)
        plt.colorbar()
        plt.title('Hough transform results')
        plt.show()
        




#%%
if __name__ == "__main__":
    points = [(0, 20), (20, 0), (40, 20)]
    img = np.zeros((60, 60, 3))

    for x, y in points:
        img[y, x] = [255, 255, 255]

    Hspace = HoughSpace(points, img.shape, 100, 100, img)
    Hspace.point_transform()
    Hspace.plot()
    print(Hspace.get_lines(3, 0))
    Hspace.draw_lines(3, 0)

    # import cv2
    # img = cv2.imread("/home/antoine/PIE/swarm-rescue/cartographer_drone_map.png")
    # g = cv2.cvtColor(img, COLOR_BGR2GRAY)
    # cv2.imshow("image base", img)
    # cv2.imshow("image gray", g)
    # print(cv2.HoughLines(g, 100, 100, 0))

    # cv2.waitKey(5000)
    # cv2.destroyAllWindows()