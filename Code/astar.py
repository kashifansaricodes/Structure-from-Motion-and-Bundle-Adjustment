import numpy as np
import matplotlib.pyplot as plt
import cv2
import heapq
from matplotlib.animation import FuncAnimation

def not_in_obstacle(node, maze):
    if node[0] < 0 or node[0] >= maze.shape[0] or node[1] < 0 or node[1] >= maze.shape[1]:
        return False
    if np.array_equal(maze[node[0], node[1]], obstacle_plot):
        return False
    
    return True

def f_value(node, goal):
    return np.sqrt((node[0] - goal[0])**2 + (node[1] - goal[1])**2)

def calculate_cost(current_cost, action, step_size=1):
    if action in [(1, 1), (-1, 1), (1, -1), (-1, -1)]:
        return current_cost + diagonal_cost
    else:
        return current_cost + cost_straight

def f_value(node, goal):
    return np.sqrt((goal[0] - node[0]) ** 2 + (goal[1] - node[1]) ** 2)

# Bidirectional A* algorithm 
def astar_bidirectional(start, goal, maze, step_size=10):
    # Initializing the open and closed sets
    open_set_start = [Node(start, None, 0, f_value(start, goal))]
    open_set_goal = [Node(goal, None, 0, f_value(goal, start))]
    start_closed_set = set()
    goal_closed_set = set()

    # Initializing the path and cost dictionaries
    path_start = {}
    path_goal = {}
    cost_start = {start: 0}
    cost_goal = {goal: 0}

    path_iteration = 0

    while open_set_start and open_set_goal:
        # Get the node with the lowest cost from the start open set
        current_node_start = heapq.heappop(open_set_start)
        start_closed_set.add(current_node_start.node)
        
        # Get the node with the lowest cost from the goal open set
        current_node_goal = heapq.heappop(open_set_goal)
        goal_closed_set.add(current_node_goal.node)

        # Check if the current nodes are in each other's closed sets
        if current_node_start.node in goal_closed_set:
            # Path found, set the intersecting node and break the loop
            intersecting_node = current_node_start.node
            break
        if current_node_goal.node in start_closed_set:
            # Path found, set the intersecting node and break the loop
            intersecting_node = current_node_goal.node
            break

        # Expand the neighbours of the current nodes
        neighbours_start = neighbour_nodes(current_node_start.node, maze, step_size)
        neighbours_goal = neighbour_nodes(current_node_goal.node, maze, step_size)


        for neighbour in neighbours_start:
            cost = calculate_cost(current_node_start.cost, neighbour)

            if neighbour in start_closed_set:
                continue

            # Check if the neighbour is in the open set
            neighbour_node = None
            for node in open_set_start:
                if node.node == neighbour:
                    neighbour_node = node
                    break

            if neighbour_node is None:
                # Add the neighbour to the open set
                neighbour_node = Node(neighbour, current_node_start, cost, f_value(neighbour, goal))
                heapq.heappush(open_set_start, neighbour_node)
            elif cost < neighbour_node.cost:
                # Update the cost of the neighbour
                neighbour_node.cost = cost
                neighbour_node.parent = current_node_start

            path_start[neighbour] = current_node_start
            cost_start[neighbour] = cost

        for neighbour in neighbours_goal:
            cost = calculate_cost(current_node_goal.cost, neighbour)

            if neighbour in goal_closed_set:
                continue

            neighbour_node = None
            for node in open_set_goal:
                if node.node == neighbour:
                    neighbour_node = node
                    break

            if neighbour_node is None:
                neighbour_node = Node(neighbour, current_node_goal, cost, f_value(neighbour, start))
                heapq.heappush(open_set_goal, neighbour_node)
            elif cost < neighbour_node.cost:
                neighbour_node.cost = cost
                neighbour_node.parent = current_node_goal

            path_goal[neighbour] = current_node_goal
            cost_goal[neighbour] = cost

    start_path = []
    goal_path = []

    node = path_start.get(intersecting_node)
    while node:
        start_path.insert(0, node.node)
        node = path_start.get(node.node)

    node = path_goal.get(intersecting_node)
    while node:
        goal_path.append(node.node)
        node = path_goal.get(node.node)

    goal_path.reverse()

    return start_path, goal_path

class Node:
    def __init__(self, node, parent, cost, f_value):
        self.node = node
        self.parent = parent
        self.cost = cost
        self.f_value = f_value
        self.total_cost = cost + f_value

    def __lt__(self, other):
        return self.total_cost < other.total_cost

def neighbour_nodes(node, maze, step_size):
    neighbours = []
    for action in actions_set:
        neighbour = (node[0] + action[0]* step_size, node[1] + action[1]* step_size)
        if not_in_obstacle(neighbour, maze):
            neighbours.append(neighbour)
    return neighbours


'''
Creating the maze
'''
obstacle_plot = np.array([255, 255, 255])

maze = np.zeros((int(3000), int(6000), 3))

circle_center = (int(4450), int(2200))   
radius_circle = int(375)
cv2.circle(maze, circle_center, radius_circle, (255, 255, 255), -1)

circle_center = (int(1120), int(2420))   
radius_circle = int(400)
cv2.circle(maze, circle_center, radius_circle, (255, 255, 255), -1)

circle_center = (int(2630), int(900))   
radius_circle = int(700)
cv2.circle(maze, circle_center, radius_circle, (255, 255, 255), -1)

'''
Cost Calculation
'''
cost_straight = 1.0
diagonal_cost = 1.4
actions_set = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1)]

start_x = int(0)
start_y = int(1500)
goal_x = int(6000)
goal_y = int(1500)

start_node = (start_y, start_x) 
goal_node = (goal_y, goal_x)
start_path, goal_path = astar_bidirectional(start_node, goal_node, maze)

path_iteration = 0
if start_path:
    print("Path found:", start_path)

    cv2.circle(maze, (start_x, start_y), 7, (255, 0, 255), -1)  
    cv2.circle(maze, (goal_x, goal_y), 7, (255, 0, 0), -1)   

    start_points = np.array(start_path)
    goal_points = np.array(goal_path)


    def update_plot(frame):
        plt.clf()
        plt.imshow(maze.astype(int))
        plt.gca().invert_yaxis()
        plt.title(f'Exploring Nodes - Frame {frame}')
        current_node = frame * 10  
        plt.plot(start_points[:current_node, 1], start_points[:current_node, 0], color='red', label='Start Path')
        plt.plot(goal_points[:current_node, 1], goal_points[:current_node, 0], color='blue', label='Goal Path')
        plt.legend()

    # Create animation
    fig, ax = plt.subplots()
    animation = FuncAnimation(fig, update_plot, frames=len(start_points) // 10, interval=100) 
    plt.show()

else:
    print("No path found.")
