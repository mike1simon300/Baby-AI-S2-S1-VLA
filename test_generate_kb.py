import random
import gymnasium as gym
import minigrid
import matplotlib.pyplot as plt
from VLA2Systems.utils import (
    print_grid,
    render_env
)
from VLA2Systems.knowledge_base import KnowledgeBase
import imageio

def save_env_image(env, filename="env_image.png"):
    """Save the environment image to a file."""
    frame = env.render()
    imageio.imwrite(filename, frame)

def main():
    # List all available MiniGrid environments
    env_list = [env_id for env_id in gym.envs.registry if "MiniGrid" in env_id]

    # Randomly select an environment
    env_name = random.choice(env_list)
    print(f"Selected Environment: {env_name}")

    # Create the environment
    env = gym.make(env_name, render_mode="rgb_array")

    # Reset the environment to initialize it
    env.reset()
    knowledge_base = KnowledgeBase(env)
    # Print the grid
    print("\nGrid Map:")
    print_grid(knowledge_base.grid_data)
    # Render the environment dynamically
    fig, ax = plt.subplots()
    render_env(env, ax, 2)
    # Save the environment image
    save_env_image(env, filename="env_image.png")

    print(f"\nKnowledge Base: \n{knowledge_base}\n")
    
    print(knowledge_base.KB)
    env.close()

if __name__ == "__main__":
    main()
