import os
import random
import yaml
import shutil
from datasets import Dataset, load_from_disk, concatenate_datasets
from VLA2Systems.task_data_generator import TaskDataGenerator
from tqdm import tqdm  # For progress bar
import contextlib
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
        self.concatinate = config.get("samples", {}).get("concatinate", True)
        self.mix_sample = config.get("samples", {}).get("mix_sample", False)
        self.mix_sample_size = config.get("samples", {}).get("mix_sample_size", 100)
        self.contribution_percentages = config.get("samples", {}).get("contribution_percentages", [])

        
        # Input text configuration
        self.include_grid = config.get("input_text", {}).get("include_grid", False)
        self.include_kb = config.get("input_text", {}).get("include_kb", True)
        self.include_mission = config.get("input_text", {}).get("include_mission", True)
        self.include_robot_location = config.get("input_text", {}).get("include_robot_location", True)
        self.include_robot_current_room = config.get("input_text", {}).get("include_robot_current_room", False)
        self.plan_prompt = config.get("input_text", {}).get("plan_prompt", "default")

        # Output text configuration
        self.include_reason = config.get("output_text", {}).get("include_reason", False)
        self.include_backward_reasoning = config.get("output_text", {}).get("include_backward_reasoning", False)
        self.include_locations = config.get("output_text", {}).get("include_locations", False)
        self.repeat_first_action = config.get("output_text", {}).get("repeat_first_action", False)
        

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
        self.generator = TaskDataGenerator(self.env_dict)
        # Ensure image save directory exists
        self.image_dir = os.path.join("datasets", self.dataset_name, "images")
        os.makedirs(self.image_dir, exist_ok=True)

    def collect_data(self, selected_difficulty="all"):
        difficulty_list = self.env_dict.items()
        difficulty_image_dir = self.image_dir
        if selected_difficulty != "all":
            difficulty_list = {selected_difficulty: self.env_dict[selected_difficulty]}.items()
            difficulty_image_dir = os.path.join(self.image_dir, selected_difficulty)
        if os.path.exists(difficulty_image_dir):
            print(f"Deleting existing images in: {difficulty_image_dir}")
            shutil.rmtree(difficulty_image_dir)  # Delete the folder and its contents

        for difficulty, env_list in difficulty_list:
            print(f"Collecting data for difficulty: {difficulty}")
            os.makedirs(os.path.join(self.image_dir, difficulty), exist_ok=True)
            sample_width = len(str(self.sample_sizes[difficulty]))
            collected_samples = 0
            with tqdm(total=self.sample_sizes[difficulty], desc=f"{difficulty} progress") as pbar:
                while collected_samples < self.sample_sizes[difficulty]:
                    env_name = random.choice(env_list)
                    seed = random.randint(0, 99999)
                    
                    self.generator.reset(difficulty=difficulty, seed=seed, env_name=env_name)
                    try:
                        plan = self.generator.generate_plan()
                    except Exception as e:
                        tqdm.write(f"[ERROR] Planning Crashed for {env_name} with seed {seed}. Skipping...")
                        plan = False
                    if plan:
                        input_text = self.generator.get_input_text(
                            include_grid=self.config.include_grid,
                            include_kb=self.config.include_kb,
                            include_mission=self.config.include_mission,
                            include_robot_location=self.config.include_robot_location,
                            include_robot_current_room=self.config.include_robot_current_room,
                            plan_prompt=self.config.plan_prompt
                        )
                        output_text = self.generator.get_output_text(
                            include_locations=self.config.include_locations,
                            include_reason=self.config.include_reason,
                            include_backward_reason=self.config.include_backward_reasoning,
                            repeat_first_action=self.config.repeat_first_action
                        )
                        
                        # Append to dataset
                        self.datasets[difficulty].append({"input": input_text, "output": output_text})
                        
                        # Save image
                        formatted_index = f"{collected_samples:0{sample_width}d}"  # Pad with leading zeros

                        img_filename = f"Image-{formatted_index}-{env_name}-seed-{seed}.png"
                        img_path = os.path.join(self.image_dir, difficulty, img_filename)
                        self.generator.save_env_image(filename=img_path)
                        
                        # Save dataset immediately
                        self.save_dataset(difficulty)
                        
                        collected_samples += 1
                        pbar.update(1)
                    else:
                        tqdm.write(f"[WARNING] Planning failed for {env_name} with seed {seed}. Retrying...")
        
        print("Data collection completed!")
        if selected_difficulty and (self.config.concatinate or self.config.mix_sample):

            # Define dataset paths
            dataset_paths = {
                "easy": f"./datasets/{self.dataset_name}/easy",
                "intermediate": f"./datasets/{self.dataset_name}/intermediate",
                "hard": f"./datasets/{self.dataset_name}/hard"
            }

        # Load datasets
        datasets = {key: load_from_disk(path) for key, path in dataset_paths.items()}
        if self.config.concatinate:
            # Concatenate datasets
            mixed_dataset = concatenate_datasets([datasets["easy"], datasets["intermediate"], datasets["hard"]])
        elif self.config.mix_sample:
            contribution_percentages = {
                "easy": self.config.contribution_percentages[0],
                "intermediate": self.config.contribution_percentages[1],
                "hard": self.config.contribution_percentages[2]
            }

            # Calculate samples per dataset
            sample_sizes = {key: int(self.config.mix_sample_size * percentage) for key, percentage in contribution_percentages.items()}
            # Sample and merge datasets
            mixed_data = []
            for key, dataset in datasets.items():
                sampled_data = dataset.shuffle(seed=42).select(range(sample_sizes[key]))  # Random sampling
                mixed_data.extend(sampled_data)

            # Convert to Hugging Face Dataset
            mixed_dataset = Dataset.from_list(mixed_data)

        # Shuffle the final dataset
        mixed_dataset = mixed_dataset.shuffle(seed=42)

        # Save the mixed dataset (optional)
        mixed_dataset.save_to_disk(f"./datasets/{self.dataset_name}/mixed")

        print("Final dataset size:", len(mixed_dataset))

    def save_dataset(self, difficulty):
        dataset_name = f"./datasets/{self.dataset_name}/{difficulty}"
        dataset = Dataset.from_list(self.datasets[difficulty])
        # Suppress tqdm interference by redirecting stdout
        # Suppress Hugging Face dataset progress bar
        with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            dataset.save_to_disk(dataset_name)
        # tqdm.write(f"Dataset for {difficulty} saved successfully!")

    def create_readme(self, config_path, selected_difficulty="all"):
        readme_name = f"README_{selected_difficulty}.md" if selected_difficulty != 'all' else "README.md"
        readme_path = os.path.join("datasets", self.dataset_name, readme_name)
        # Calculate the dataset size in MB
        dataset_name = f"./datasets/{self.dataset_name}"
        with open(readme_path, 'w') as f:
            f.write(f"Dataset Name: {self.dataset_name}\n")
            f.write(f"Generated on: {self.get_timestamp()}\n")
            f.write(f"Dataset size per difficulty:\n")
            for diff, size in self.sample_sizes.items():
                f.write(f"  {diff}: {size} samples\n")
            f.write("\nConfiguration File:\n")
            with open(config_path, 'r') as config_file:
                f.write(config_file.read())
        print(f"README file saved as {readme_name}")
    
    @staticmethod
    def get_timestamp():
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Example Usage:
# collector = DataCollector("config.yaml")
# collector.collect_data()
