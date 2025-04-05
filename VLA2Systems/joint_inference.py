

import os
import yaml
import torch
import numpy
import random
from copy import deepcopy
from gymnasium.core import Env
import matplotlib.pyplot as plt
from datasets import load_from_disk
from IPython.display import display, clear_output
from transformers import AutoModelForCausalLM, AutoTokenizer
from VLA2Systems.task_data_generator import TaskDataGenerator
import VLA2Systems.rl_utils as utils
from VLA2Systems.utils import generate_verifier


class JointInferenceConfigParser:
    def __init__(self, config_path):
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        self.system2 = config.get("system2")
        self.system1 = config.get("system1")
        # System 2 settings
        self.model_path = self.system2.get("model_path", "")
        self.cache_dir = self.system2.get("cache_dir", "")
        self.from_prompt = self.system2.get("from_prompt", False)
        self.dataset_path = self.system2.get("dataset_path", "")
        self.temperature = self.system2.get("temperature", 0.7)
        self.top_p = self.system2.get("top_p", 0.9)

        # System 1 settings
        self.s1_model_path = self.system1.get("model_path", "")
        self.argmax = self.system1.get("argmax", False)
        self.pause = self.system1.get("pause", 0.05)
        self.text = self.system1.get("text", True)
        self.memory = self.system1.get("memory", False)
        self.visulize = self.system1.get("visulize", True)
        self.verify = self.system1.get("verify", True)

class System2Inference:
    def __init__(self, config_path, env=None):
        self.config = JointInferenceConfigParser(config_path)
        
        if not os.path.exists(self.config.model_path):
            raise ValueError("[System2Inference] model_path is not valid")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_path,
            cache_dir=self.config.cache_dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_path,
            device_map="auto",
            torch_dtype=torch.float16,
            cache_dir=self.config.cache_dir
        )
        if os.path.exists(self.config.dataset_path):
            self.dataset = load_from_disk(self.config.dataset_path)
        else:
            self.dataset = None
        self.model.eval()
        if torch.cuda.is_available():
            self.model.to("cuda")
        self.env = env
        self.generator = None
        if env is not None:
            self.init_env(env)
            

    def generate_text(self, prompt, max_length=4096):
        appended_prompt = f"### Instruction:\n{prompt}\n\n### Response:\n"
        inputs = self.tokenizer(appended_prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        output = self.model.generate(
            **inputs,
            max_length=max_length,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    def update_env(self, env):
        self.env = env
        self.generator.init_planner(self.env, self.generator.rest_variables)

    def init_env(self, env, seed=None):
        self.env = env
        self.generator = TaskDataGenerator([], env=env)
        self.generator.reset(seed=seed)

    def __call__(self, *args, **kwds):
        if len(args) == 0 and self.dataset:
            index = random.randint(0, len(self.dataset)-1)
            example = self.dataset[index]
            prompt = example["input"]
            return self.generate_text(prompt)
        if len(args) == 0 and self.dataset:
            index = random.randint(0, len(self.dataset)-1)
            example = self.dataset[index]
            prompt = example["input"]
            return self.generate_text(prompt)
        elif len(args)>0 and self.config.from_prompt:
            prompt = args[0]
            return self.generate_text(prompt)
        elif len(args)>0 and isinstance(args[0], int):
            example = self.dataset[args[0]]
            prompt = example["input"]
            return self.generate_text(prompt)
        elif len(args)>0 and isinstance(args[0], Env):
            self.update_env(args[0])
            prompt = self.generator.get_input_text(include_all=True)
            # print(" THE INPUT IS: \n", prompt, " \n END OF INPUT")
            output = self.generate_text(prompt)
            # print(" THE OUTPUT IS: \n", output, " \n END OF OUTPUT")
            return output

    def get_input(self, env):
        self.update_env(env)
        prompt = self.generator.get_input_text(include_all=True)
        appended_prompt = f"### Instruction:\n{prompt}\n\n### Response:\n"
        return appended_prompt
    
class System1Inferece:
    def __init__(self, config_path, env):
        self.config = JointInferenceConfigParser(config_path)
        self.env = env
        if not os.path.exists(self.config.s1_model_path):
            raise ValueError("[System1Inference] model_path is not valid")
        self.agent = utils.Agent(self.env.observation_space, 
                                 self.env.action_space, 
                                 self.config.s1_model_path,
                                 argmax=self.config.argmax, 
                                 use_memory=self.config.memory, 
                                 use_text=self.config.text)

    def step(self, sub_task, obs=None):
        if obs is None:
            obs = deepcopy(self.obs)
        obs['mission'] = sub_task
        action = self.agent.get_action(obs)
        obs, reward, terminated, truncated, _ = self.env.step(action)
        if self.config.verify:
            terminated, reward = self.verify_sub_task(action)
        self.obs = obs
        done = terminated | truncated
        self.agent.analyze_feedback(reward, done)
        if self.config.visulize:
            image = self.env.render()
            self.show_frame_Jupyter(image)
            print(sub_task)
        return done

    def step_for(self, steps, sub_task, obs=None):
        if obs is None:
            obs = self.obs
        for i in range(steps):
            self.step(sub_task)

    def step_untill(self, sub_task, obs=None, max_steps=1000):
        if obs is None:
            obs = self.obs
        step_counter = 0
        done = False
        while not done and step_counter < max_steps:
            done = self.step(sub_task)
            print(done)


    def show_frame_Jupyter(self, image):
        """
        Displays a single frame in a Jupyter Notebook.
        
        :param image: The image to display (e.g., from env.render()).
        """
        clear_output(wait=True)  # Clear previous output
        plt.imshow(image)
        plt.axis("off")  # Hide axes
        display(plt.gcf())  # Show figure

    def init_sub_task(self, sub_task, env=None):
        if env is None:
            env = self.env
        if self.config.verify:
            self.instrs = generate_verifier(sub_task)
            if not self.instrs:
                self.instrs = None
                return None
            self.instrs.reset_verifier(env.unwrapped)
            return self.instrs
        return True
    # def init_verifier(self, sub_task, env=None):
    #     if env is None:
    #         env = self.env
    #     params = self.parse_mission(sub_task)
    #     verifierClass, obj_desc = self.shape_verifier(params)
    #     self.verifer = verifierClass(obj_desc: ObjDesc, env)

    def verify_sub_task(self, action):
        # If we've successfully completed the mission
        status = self.instrs.verify(action)
        terminated = False
        reward = 0
        if status == "success":
            terminated = True
            reward = 1.0
        elif status == "failure":
            terminated = True
            reward = 0
        return terminated, reward
