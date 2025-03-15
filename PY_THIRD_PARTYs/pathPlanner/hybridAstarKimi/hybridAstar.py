import numpy as np
import matplotlib.pyplot as plt
from heapq import heappush, heappop
from math import sin, cos, atan2, sqrt, pi

# Define vehicle parameters
WHEEL_BASE = 2.5  # Vehicle wheelbase (m)
MIN_TURNING_RADIUS = 5.0  # Minimum turning radius (m)
VEHICLE_LENGTH = 4.0  # Vehicle length (m)
VEHICLE_WIDTH = 2.0  # Vehicle width (m)

# Define the environment
OBSTACLES = [(2, 2, 4, 4), (8, 8, 4, 4)]  # (x, y, width, height)
START = (0, 0, 0)  # (x, y, theta)
GOAL = (10, 10, 0)  # (x, y, theta)
GRID_RESOLUTION = 0.5  # Grid resolution (m)

# Define motion primitives (Dubins path segments)
def generate_motion_primitives():
    primitives = []
    for steering in [-1, 0, 1]:  # Left, Straight, Right
        for length in [1.0, 2.0, 3.0]:  # Path length
            primitives.append((steering, length))
    return primitives

MOTION_PRIMITIVES = generate_motion_primitives()

# Check collision with obstacles
def is_collision(x, y, theta, obstacles):
    for ox, oy, ow, oh in obstacles:
        if (ox <= x <= ox + ow and oy <= y <= oy + oh):
            return True
    return False

# Compute Dubins path
def dubins_path(x, y, theta, steering, length):
    if steering == -1:  # Left turn
        radius = MIN_TURNING_RADIUS
        angle = length / radius
        x_new = x + radius * (cos(theta + angle) - cos(theta))
        y_new = y + radius * (sin(theta + angle) - sin(theta))
        theta_new = (theta + angle) % (2 * pi)
    elif steering == 0:  # Straight
        x_new = x + length * cos(theta)
        y_new = y + length * sin(theta)
        theta_new = theta
    elif steering == 1:  # Right turn
        radius = MIN_TURNING_RADIUS
        angle = length / radius
        x_new = x + radius * (cos(theta - angle) - cos(theta))
        y_new = y + radius * (sin(theta - angle) - sin(theta))
        theta_new = (theta - angle) % (2 * pi)
    return x_new, y_new, theta_new

# Hybrid A* algorithm
def hybrid_a_star(start, goal, obstacles, grid_resolution):
    open_set = []
    heappush(open_set, (0, start))
    came_from = {}
    cost_so_far = {start: 0}
    visited = set()

    while open_set:
        _, current = heappop(open_set)
        if current in visited:
            continue
        visited.add(current)

        # Visualize current node
        plt.plot(current[0], current[1], 'bo', markersize=4)

        if sqrt((current[0] - goal[0]) ** 2 + (current[1] - goal[1]) ** 2) < grid_resolution:
            return reconstruct_path(came_from, current)

        for steering, length in MOTION_PRIMITIVES:
            x_new, y_new, theta_new = dubins_path(current[0], current[1], current[2], steering, length)
            new_node = (x_new, y_new, theta_new)

            if is_collision(x_new, y_new, theta_new, obstacles):
                continue

            new_cost = cost_so_far[current] + length
            if new_node not in cost_so_far or new_cost < cost_so_far[new_node]:
                cost_so_far[new_node] = new_cost
                priority = new_cost + sqrt((new_node[0] - goal[0]) ** 2 + (new_node[1] - goal[1]) ** 2)
                heappush(open_set, (priority, new_node))
                came_from[new_node] = current

    return None

# Reconstruct path from start to goal
def reconstruct_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path

# Visualize the path
def visualize_path(path):
    x, y = zip(*[(node[0], node[1]) for node in path])
    plt.plot(x, y, 'r--', linewidth=2)
    plt.plot(path[0][0], path[0][1], 'go', markersize=8)  # Start
    plt.plot(path[-1][0], path[-1][1], 'ro', markersize=8)  # Goal

# Main function
def main():
    plt.figure(figsize=(10, 10))
    plt.grid(True)
    plt.axis('equal')
    plt.xlim(-1, 15)
    plt.ylim(-1, 15)

    # Draw obstacles
    for ox, oy, ow, oh in OBSTACLES:
        rect = plt.Rectangle((ox, oy), ow, oh, color='k')
        plt.gca().add_patch(rect)

    # Run Hybrid A* algorithm
    path = hybrid_a_star(START, GOAL, OBSTACLES, GRID_RESOLUTION)
    if path:
        visualize_path(path)
    else:
        print("No path found")

    plt.show()

if __name__ == "__main__":
    main()