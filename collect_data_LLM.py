import os
import argparse
import yaml
import time
from VLA2Systems.task_data_collector import DataCollector


def get_dataset_size(dataset_name):
    """Calculate total dataset size in bytes."""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(dataset_name):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size / (1024 * 1024)  # Convert to MB


def write_readme(dataset_name, config_path):
    """Create README.md inside the dataset folder with config details and dataset size."""
    with open(config_path, 'r') as file:
        config_content = file.read()
    
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    dataset_size = get_dataset_size(dataset_name)
    
    readme_content = f"""
    # Dataset: {dataset_name}
    
    **Generated on:** {timestamp}
    
    **Dataset Size:** {dataset_size:.2f} MB
    
    ## Configuration:
    
    ```yaml
    {config_content}
    ```
    """
    
    readme_path = os.path.join("datasets", dataset_name, "README.md")
    with open(readme_path, 'w') as readme_file:
        readme_file.write(readme_content)
    print("README.md created successfully!")


def main():
    parser = argparse.ArgumentParser(description="Collect data for LLM fine-tuning based on mission planning.")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration YAML file")
    parser.add_argument("--difficulty", type=str, default="all", choices=["all", "easy", "intermediate", "hard"],
                        help="Difficulty level to generate (default: all)")

    args = parser.parse_args()

    collector = DataCollector(args.config)
    collector.collect_data(args.difficulty)    
    collector.create_readme(args.config, args.difficulty)


if __name__ == "__main__":
    main()
