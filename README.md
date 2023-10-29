# CIREX Project
Code and algorithms for the CIREX Project, organized by the French DGA (Direction Générale de l'Armement). This code is made to work on a custom environment made by the DGA, based on [Simple Playgrounds, an environment made by gaorkl](https://github.com/gaorkl/simple-playgrounds). 

The main files are:
* `my_final_drone.py`: main class of the project, containing the whole behavior of the drones
* `algorithme_astar.py`: path-finding algorithm A* implementation
* `KHT.py`: "Kernel-Based Hough Transform", an algorithm used for line detection in cartography

## Features
* A custom-made swarm behavior, based on communication, LIDAR detection and roles
* A system of queue of drones, with a leader and followers, all sharing information between them
* A custom-made algorithm for line detection in cartography, using the lidar data

## Screenshots
![cirex_gif](https://github.com/remsto/PIE_CIREX/blob/master/gif/cirex_gif.gif)



