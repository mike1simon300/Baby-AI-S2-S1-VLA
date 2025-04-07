import random
import gymnasium as gym
import minigrid
import matplotlib.pyplot as plt
from VLA2Systems.utils import render_env, get_grid_text
from VLA2Systems.knowledge_base import KnowledgeBase
from VLA2Systems.task_planning import RobotPlanner
import imageio

class TaskDataGenerator:
    def __init__(self, env_options, seed=None, env=None, verbose=0):
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
        self.from_env = False
        self.verbose = verbose
        if env is not None:
            self.from_env = True
            self.env = env

    def reset(self, seed=None, difficulty=None, env_name=None):
        if seed is None:
            seed = self.seed
        if self.from_env:
            self.rest_variables = self.env.reset(seed=seed)
        elif env_name is not None or difficulty is not None or self.env is None:
            env_name = self.select_environment(difficulty, env_name)
            self.env = gym.make(env_name, render_mode="rgb_array")
        self.rest_variables = self.env.reset(seed=seed)
        self.init_planner(self.env, self.rest_variables)
        return self.rest_variables
        # print(f"Initialized environment with seed {seed}")

    def init_planner(self, env=None, rest_variables=None):
        if env is None:
            env = self.env
        if rest_variables is None:
            rest_variables = self.rest_variables
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
            verbose=self.verbose
        )
        self.mission = self.rest_variables[0]["mission"]


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

    def generate_backward_logic(self, plan):
        """
        Generates a backward reasoning paragraph based on the provided plan.
        
        Each plan entry is a tuple:
        (action, object_type, color, position, room_id, reason_msg)
        
        The reasoning starts from the goal (the last step) and works backward,
        connecting the steps with phrases like "but" and ending with the robot's current room.
        
        Parameters:
        plan (list of tuples): The task plan from start to finish.
        robot_current_room (int): The current room of the robot.
        
        Returns:
        str: The backward reasoning paragraph.
        """
        if not plan:
            return ""
        robot_current_room = self.planner.get_current_room(self.start_location)
        # doors = generator.knowledge_base.KB['connections']
        # Start from the goal: the last step's reason message
        
        backward_text = plan[-1][-1].strip()
        
        # If there are prior steps, add them in reverse order
        if len(plan) > 1:
            # Reverse the plan excluding the last (goal) step.
            backward_text = "Perform these steps to " + backward_text
            reversed_steps = plan[:-1][::-1]
            
            # If more than one step, join them with commas and add a "but" before the last one.
            if len(reversed_steps) == 1:
                backward_text += ", " + reversed_steps[0][-1].strip()
            else:
                for i, step in enumerate(reversed_steps):
                    # Insert a connector: for the first step in the reverse order, just a comma
                    if i == 0:
                        backward_text += ", " + step[-1].strip()
                    # For the last one in the reversed sequence, use "but" to emphasize the need to satisfy this condition
                    elif i == len(reversed_steps) - 1:
                        backward_text += ", but " + step[-1].strip()
                    else:
                        backward_text += ", " + step[-1].strip()
        
        # Append an ending phrase indicating the robot's current location.
        backward_text += f", which is where the robot is currently in room {robot_current_room}."
        return backward_text
    
    def visualize(self):
        fig, ax = plt.subplots()
        render_env(self.env, ax, 2)
        plt.show()
    
    def save_env_image(self, filename="env_image.png"):
        frame = self.env.render()
        imageio.imwrite(filename, frame)

    # def plan2text(self, include_kb=False, plan=None, print_plan=False, 
    #               include_mission=False, include_robot_location=False,
    #               include_all=False, include_locations=False, 
    #               include_reason=False, include_backward_reason=False):
    #     if plan is None:
    #         plan = self.plan
    #     if plan is None:
    #         raise Exception("Plan is not there")
    #     text = ""
    #     if include_kb or include_all:
    #         text += "Knowledge Base:\n" + str(self.knowledge_base)
    #     if include_robot_location or include_all:
    #         text += "\nRobot location: " + str(self.start_position)
    #     if include_mission or include_all:
    #         text += "\nMission: " + self.mission
    #     plan_text = str(self.planner.__str__(plan))
    #     text += "\nPlan is: \n" + plan_text
    #     if print_plan:
    #         print(text)
    #     return text

    def plan2text(self, plan=None, include_locations=False, 
                  include_reason=False, include_step=True):
        if plan is None:
            plan = self.plan
        steps = []
        for i, action in enumerate(plan, start=1):
            if not action:  # Skip if the action is empty
                continue

            action_type = action[0]
            color = action[2]
            if color == "":
                color = random.choice(["a","the"])
            loc = ""
            if include_locations and action_type != 'drop_next_to' and \
                len(action) > 3 and action[3] is not None:
                loc = action[3]
            elif include_locations and action_type == 'drop_next_to' and \
                len(action) > 5 and action[5] is not None:
                loc = action[5]
            
            loc = f' at {loc}' if loc != "" else loc
            reason = ""
            step = ""
            if include_step:
                step = f"Step {i}: "
            if include_reason:
                reason = f". Reason is: {action[-1]}"
            if action_type == 'go_to' and len(action) > 2:
                steps.append(f"{step}Go to {color} {action[1]}{loc}{reason}")
            elif action_type == 'pick_up' and len(action) > 2:
                steps.append(f"{step}Pick up {color} {action[1]}{loc}{reason}")
            elif action_type == 'open' and len(action) > 0:
                steps.append(f"{step}Open {color} {action[1]}{loc}{reason}")
            elif action_type == 'drop_next_to' and len(action) > 0:
                color2 = action[4]
                if color2 == "":
                    color = random.choice(["a","the"])
                steps.append(
                    f"{step}Drop {color} {action[1]} "+
                    f"next to {color2} {action[3]}{loc}{reason}")
            else:
                steps.append(f"{step}Invalid action {action}")

        return "\n".join(steps)


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
            text += ", which is in Room " + str(self.planner.get_current_room(self.start_location))
        if include_mission or include_all:
            text += "\nMission: " + self.mission
        if plan_prompt == "default":
            text += "\nIn order to achieve the mission the robot should preform these steps:\n"
        else:
            text += plan_prompt
        return text

    def get_output_text(self, plan=None, include_locations=False, 
                        include_reason=False, include_backward_reason=False, 
                        repeat_first_action=False, include_all=False):
        plan_text = ""
        if plan is None:
            plan = self.plan
        if plan is None:
            raise Exception("Plan is not there")
        if include_all:
            include_locations=True
            include_reason=True
            include_backward_reason=True
            repeat_first_action = True
        if include_backward_reason:
            plan_text += "Reasoning Stage: \n" +\
                  self.generate_backward_logic(plan) +\
                    "\nEnd of Reasoning Stage\nExecution Stage:\n"
        plan_text += self.plan2text(plan, include_locations, include_reason)
        if repeat_first_action:
            first_action = "\nSo the task the robot should do now is: \n" +\
                self.plan2text(plan, include_locations, include_reason, 
                               include_step=False).split('\n')[0]
            if include_reason:
                first_action = first_action.split('Reason')[0]
            plan_text += first_action
        return plan_text
