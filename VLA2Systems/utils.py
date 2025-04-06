
import gymnasium as gym
import minigrid
import matplotlib.pyplot as plt
import minigrid.core
import minigrid.core.grid
import minigrid.minigrid_env
from minigrid.envs.babyai.core.verifier import (
    ObjDesc,
    OpenInstr,
    GoToInstr,
    PickupInstr,
    PutNextInstr
)
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

def parse_mission(mission):
    """Extracts task details from the environment mission description."""
    mission = mission.lower()
    words = mission.split()
    
    task_type = None
    obj1, color1 = None, None
    obj2, color2 = None, None
    
    if "pick up" in mission:
        task_type = "pick_up"
    elif "go to" in mission or "get to" in mission:
        task_type = "go_to"
    elif "open" in mission and "door" in mission:
        task_type = "open_door"
    elif "put" in mission and "next to" in mission:
        task_type = "put_next_to"
    
    colors = {"red", "green", "blue", "purple", "yellow", "grey"}
    objects = {"ball", "box", "door", "key", "goal"}
    color1, obj1, color2, obj2 = "", "", "", ""
    for i, word in enumerate(words):
        if word in colors:
            if obj1 == "":
                color1 = word
            else:
                color2 = word
        elif word in objects:
            if obj1 == "":
                obj1 = word
            else:
                obj2 = word
    
    return task_type, color1, obj1, color2, obj2

def generate_verifier(mission):
    task_type, color1, obj1, color2, obj2 = parse_mission(mission)
    if color1 == "":
        color1 = None
    if color2 == "":
        color2 = None
    # print(f"Task parsed: {task_type}, {color1}, {obj1}, {color2}, {obj2}")
    if task_type == "pick_up":
        instrs = PickupInstr(ObjDesc(obj1, color1))
    elif task_type == "go_to":
        instrs = GoToInstr(ObjDesc(obj1, color1))
    elif task_type == "open_door":
        instrs = OpenInstr(ObjDesc(obj1, color1))
    elif task_type == "put_next_to":
        instrs = PutNextInstr(ObjDesc(obj1, color1), 
                              ObjDesc(obj2, color2))
    else:
        print("No valid task detected.")
        return False
    return instrs
