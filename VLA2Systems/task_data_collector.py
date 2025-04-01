import os
import random
import yaml
import shutil
from datasets import Dataset, DatasetDict
from VLA2Systems.task_data_generator import TaskDataGenerator
from tqdm import tqdm  # For progress bar

import yaml

class DataCollectorConfigParser:
    def __init__(self, config_path):
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        
        # General settings
        self.dataset_name = config.get("dataset_name", "default_dataset")
        
        # Environment lists for difficulties
        self.easy_envs = config.get("environments", {}).get("easy", [])
        self.intermediate_envs = config.get("environments", {}).get("intermediate", [])
        self.hard_envs = config.get("environments", {}).get("hard", [])
        
        # Number of samples per difficulty
        self.easy_samples = config.get("samples", {}).get("easy", 100)
        self.intermediate_samples = config.get("samples", {}).get("intermediate", 100)
        self.hard_samples = config.get("samples", {}).get("hard", 100)
        
        # Input text configuration
        self.include_grid = config.get("input_text", {}).get("include_grid", False)
        self.include_kb = config.get("input_text", {}).get("include_kb", True)
        self.include_mission = config.get("input_text", {}).get("include_mission", True)
        self.include_robot_location = config.get("input_text", {}).get("include_robot_location", True)
        self.include_robot_current_room = config.get("input_text", {}).get("include_robot_current_room", False)
        self.plan_prompt = config.get("input_text", {}).get("plan_prompt", "default")


class DataCollector:
    def __init__(self, config_path):
        # Load and parse the config
        self.config = DataCollectorConfigParser(config_path)
        self.dataset_name = self.config.dataset_name
        self.env_dict = {
            "easy": self.config.easy_envs,
            "intermediate": self.config.intermediate_envs,
            "hard": self.config.hard_envs
        }
        self.sample_sizes = {
            "easy": self.config.easy_samples,
            "intermediate": self.config.intermediate_samples,
            "hard": self.config.hard_samples
        }
        self.datasets = {"easy": [], "intermediate": [], "hard": []}
        
        # Ensure image save directory exists
        self.image_dir = os.path.join("datasets", self.dataset_name, "images")
        os.makedirs(self.image_dir, exist_ok=True)

    def collect_data(self):
        for difficulty, env_list in self.env_dict.items():
            print(f"Collecting data for difficulty: {difficulty}")
            os.makedirs(os.path.join(self.image_dir, difficulty), exist_ok=True)
            
            for index in tqdm(range(self.sample_sizes[difficulty]), desc=f"{difficulty} progress"):
                env_name = random.choice(env_list)
                seed = random.randint(0, 9999)
                
                generator = TaskDataGenerator(env_dict={difficulty: [env_name]})
                generator.reset(difficulty=difficulty, seed=seed)
                
                plan = generator.generate_plan()
                if plan:
                    input_text = generator.get_input_text(
                        include_grid=self.config.include_grid,
                        include_kb=self.config.include_kb,
                        include_mission=self.config.include_mission,
                        include_robot_location=self.config.include_robot_location,
                        include_robot_current_room=self.config.include_robot_current_room,
                        plan_prompt=self.config.plan_prompt
                    )
                    output_text = generator.get_output_text()
                    
                    # Append to dataset
                    self.datasets[difficulty].append({"input": input_text, "output": output_text})
                    
                    # Save image
                    img_filename = f"Image-{index}-{env_name}-{seed}.png"
                    img_path = os.path.join(self.image_dir, difficulty, img_filename)
                    generator.save_env_image(filename=img_path)
                    
                    # Save dataset immediately
                    self.save_dataset(difficulty)
                else:
                    print(f"[WARNING] Planning failed for {env_name} with seed {seed}.")
        
        print("Data collection completed!")

    def save_dataset(self, difficulty):
        dataset_name = f"./datasets/{self.dataset_name}/{difficulty}"
        dataset = Dataset.from_list(self.datasets[difficulty])
        dataset.save_to_disk(dataset_name)
        print(f"Dataset for {difficulty} saved successfully!")

# Example Usage:
# collector = DataCollector("config.yaml")
# collector.collect_data()
