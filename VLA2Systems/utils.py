
import gymnasium as gym
import minigrid
import matplotlib.pyplot as plt
import minigrid.core
import minigrid.core.grid
import minigrid.minigrid_env
import minigrid.minigrid_env

def print_grid(grid_data):
    """Print the grid in a readable format."""
    for row in grid_data:
        print("".join(cell.type[0].upper() if cell else '.' for cell in row))

def get_grid_text(grid_data, robot_location):
    """Print the grid in a readable format."""
    text = ""
    robot_col, robot_row = robot_location[1]
    for i, row in enumerate(grid_data):
        line = "".join(cell.type[0].upper() if cell else '.' for cell in row)
        if i == robot_row:
            line = line[:robot_col] + 'R' + line[robot_col + 1:]
        text += line + "\n"
    return text

def render_env(env, ax, pause=0.001, iteration:str|int=""):
    # Get the RGB image from the MiniGrid environment
    frame = env.render()
    ax.clear()  # Clear the previous frame
    ax.imshow(frame)  # Display the new frame
    ax.axis('off')
    ax.set_title(f"Iteration {iteration}")  # Title with iteration number
    plt.draw()
    plt.pause(pause)  # Pause for 100ms before updating
