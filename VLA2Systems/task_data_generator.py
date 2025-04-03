import random
import gymnasium as gym
import minigrid
import matplotlib.pyplot as plt
from VLA2Systems.utils import render_env, get_grid_text
from VLA2Systems.knowledge_base import KnowledgeBase
from VLA2Systems.task_planning import RobotPlanner
import imageio

class TaskDataGenerator:
    def __init__(self, env_options, seed=None):
        """
        Initialize the data generator.
        :param env_options: Either a list of environment names or a dictionary categorized by difficulty.
        :param seed: Optional seed for reproducibility.
        """
        self.env_options = env_options
        self.seed = seed if seed is not None else random.randint(1, 1000)
        self.env_name = None
        self.env = None
        self.knowledge_base = None
        self.planner = None
        self.plan = None
    
    def reset(self, seed=None, difficulty=None, env_name=None):
        if seed is None:
            seed = self.seed
        if env_name is not None or difficulty is not None or self.env is None:
            env_name = self.select_environment(difficulty, env_name)
            self.env = gym.make(env_name, render_mode="rgb_array")
        self.rest_variables = self.env.reset(seed=seed)
        self.knowledge_base = KnowledgeBase(self.env)
        self.robot_position = self.env.unwrapped.agent_pos
        self.robot_direction = self.env.unwrapped.agent_dir
        self.start_location = (
            int(self.robot_direction),
            (int(self.robot_position[0]), int(self.robot_position[1]))
        )
        if self.env.unwrapped.carrying is None:
            self.holding = None
        else:
            self.holding = (self.env.unwrapped.carrying.type, 
                            self.env.unwrapped.carrying.color)

        self.start_position = self.start_location[1]
        self.planner = RobotPlanner(
            self.knowledge_base,
            start_location=self.start_location,
            holding=self.holding,
            verbose=0
        )
        self.mission = self.rest_variables[0]["mission"]
        # print(f"Initialized environment with seed {seed}")

    def select_environment(self, difficulty=None, env_name=None):
        """
        Select an environment based on the given difficulty or randomly.
        :param difficulty: Optional difficulty level if env_options is a dictionary.
        """
        if env_name:
            self.env_name = env_name
        elif isinstance(self.env_options, dict):  # If categorized by difficulty
            if difficulty and difficulty in self.env_options:
                self.env_name = random.choice(self.env_options[difficulty])
            else:
                all_envs = [env for env_list in self.env_options.values() for env in env_list]
                self.env_name = random.choice(all_envs)
        elif isinstance(self.env_options, list):  # If just a list
            self.env_name = random.choice(self.env_options)
        elif isinstance(self.env_options, str):  # If just a list
            self.env_name = self.env_options
        else:
            raise ValueError("Invalid environment options format. Provide a string or list or a dictionary of lists.")
        
        # print(f"Selected Environment: {self.env_name}")
        return self.env_name

    def parse_mission(self):
        """Extracts task details from the environment mission description."""
        mission = self.mission.lower()
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

    def generate_plan(self):
        task_type, color1, obj1, color2, obj2 = self.parse_mission()
        # print(f"Task parsed: {task_type}, {color1}, {obj1}, {color2}, {obj2}")
        if task_type == "pick_up":
            plan = self.planner.plan_pick_up(color1, obj1)
        elif task_type == "go_to":
            plan = self.planner.plan_go_to(color1, obj1)
        elif task_type == "open_door":
            plan = self.planner.plan_open_door(color1)
        elif task_type == "put_next_to":
            plan = self.planner.plan_drop_next_to(color1, obj1, color2, obj2)
        else:
            print("No valid task detected.")
            return False
        self.plan = plan
        return plan
        
    def visualize(self):
        fig, ax = plt.subplots()
        render_env(self.env, ax, 2)
        plt.show()
    
    def save_env_image(self, filename="env_image.png"):
        frame = self.env.render()
        imageio.imwrite(filename, frame)

    def plan2text(self, include_kb=False, plan=None, print_plan=False, 
                  include_mission=False, include_robot_location=False,
                  include_all=False):
        if plan is None:
            plan = self.plan
        if plan is None:
            raise Exception("Plan is not there")
        text = ""
        if include_kb or include_all:
            text += "Knowledge Base:\n" + str(self.knowledge_base)
        if include_robot_location or include_all:
            text += "\nRobot location: " + str(self.start_position)
        if include_mission or include_all:
            text += "\nMission: " + self.mission
        plan_text = str(self.planner.__str__(plan))
        text += "\nPlan is: \n" + plan_text
        if print_plan:
            print(text)
        return text

    def get_input_text(self, include_grid=False, include_kb=True, include_mission=True, 
                       include_robot_location=True, include_robot_current_room=False, 
                       plan_prompt="default", include_all=False):
        text = ""
        if include_grid or include_all:
            text += "Grid Map of the environment:\n" + get_grid_text(self.knowledge_base.grid_data, 
                                                                     self.start_location)
        if include_kb or include_all:
            text += "Knowledge Base:\n" + str(self.knowledge_base)
        if include_robot_location or include_all:
            text += "\nRobot location: " + str(self.start_position)
        if include_robot_current_room or include_all:
            text += ", which is in Room " + str(self.planner.current_room(self.start_location))
        if include_mission or include_all:
            text += "\nMission: " + self.mission
        if plan_prompt == "default":
            text += "\nIn order to achieve the mission the robot should preform these steps:\n"
        else:
            text += plan_prompt
        return text

    def get_output_text(self, plan=None):
        text = ""
        if plan is None:
            plan = self.plan
        if plan is None:
            raise Exception("Plan is not there")
        plan_text = str(self.planner.__str__(plan))
        return plan_text
