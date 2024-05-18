import matplotlib.pyplot as plt
import numpy as np

# Actual figure dimensions
canvas_width = 6000
canvas_height = 3000

# Downscale factor for visualization
scale_factor = 10

# Compute downscaled dimensions
display_width = canvas_width // scale_factor
display_height = canvas_height // scale_factor

# Create a figure with downscaled size
fig, ax = plt.subplots(figsize=(display_width/100, display_height/100), dpi=100)

# Set axis limits based on actual dimensions
ax.set_xlim(0, canvas_width)
ax.set_ylim(0, canvas_height)

# Set axis ticks based on actual dimensions
ax.set_xticks(range(0, canvas_width + 1, 1000))
ax.set_yticks(range(0, canvas_height + 1, 500))

# Set figure title
fig.suptitle(f'Figure ({canvas_width} x {canvas_height})')

# Define obstacle details
obstacles = [
    {'center': (1120, 2425), 'radius': 400},
    {'center': (2630, 900), 'radius': 700},
    {'center': (4450, 2200), 'radius': 375}
]

# Draw circles for obstacles
for obstacle in obstacles:
    circle = plt.Circle(obstacle['center'], obstacle['radius'], color='red')
    ax.add_patch(circle)

# Define start and goal points
start_point = {'x': 0, 'y': 1500}
goal_point = {'x': 6000, 'y': 1500}

# Plot start and goal points
plt.plot(start_point['x'], start_point['y'], 'rD', markersize=5)
plt.plot(goal_point['x'], goal_point['y'], 'rD', markersize=5)

# Tunable variables
robot_charge = 1.0
goal_charge = 5.0
obstacle_charge = 100.0
step_size = 100
goal_threshold = 100
obstacle_threshold = 200

'''
Force Calculations
'''
def compute_attractive_force(robot_position, goal_position):
    direction_vector = goal_position - robot_position
    distance = np.linalg.norm(direction_vector)
    if distance <= goal_threshold:
        return np.zeros(2)
    force_magnitude = goal_charge * robot_charge / (distance ** 2)
    return force_magnitude * direction_vector / distance

def compute_repulsive_force(robot_position, obstacle_position, obstacle_radius):
    direction_vector = robot_position - obstacle_position
    distance = np.linalg.norm(direction_vector)
    if distance <= obstacle_radius + obstacle_threshold:
        if distance <= obstacle_radius:
            distance = obstacle_radius
        force_magnitude = obstacle_charge * robot_charge / (distance ** 2)
        return force_magnitude * direction_vector / distance
    return np.zeros(2)

'''
Planning Algorithm
'''
robot_position = np.array([start_point['x'], start_point['y']], dtype=float)
trajectory = [robot_position]

max_steps = 1000
step_count = 0

while np.linalg.norm(robot_position - np.array([goal_point['x'], goal_point['y']])) > goal_threshold and step_count < max_steps:
    # Calculate attractive force
    attractive_force = compute_attractive_force(robot_position, np.array([goal_point['x'], goal_point['y']]))
    
    # Calculate repulsive forces
    total_repulsive_force = np.zeros(2)
    for obs in obstacles:
        total_repulsive_force += compute_repulsive_force(robot_position, np.array(obs['center']), obs['radius'])
    
    # Calculate resultant force
    total_force = attractive_force + total_repulsive_force
    
    # Normalize resultant force
    if np.linalg.norm(total_force) > 0:
        total_force /= np.linalg.norm(total_force)
    
    # Move robot and update path
    new_position = robot_position + step_size * total_force
    trajectory.append(new_position)
    
    # Update robot position
    robot_position = new_position
    
    step_count += 1

# Plot the final path
path_x, path_y = zip(*trajectory)
plt.plot(path_x, path_y, 'g-', linewidth=3)

# Print nodes in the final path
print("\nNodes in the Final Path:")
for i, node in enumerate(trajectory):
    print(f"Node {i+1}: ({node[0]:.2f}, {node[1]:.2f})")

# Display the plot
plt.tight_layout()
plt.show()
