import gymnasium as gym
import minigrid
from minigrid.envs import DoorKeyEnv
import numpy as np
import matplotlib.pyplot as plt
import time

# Your make_env function
def make_env(env_key, seed=None, render_mode=None):
    env = gym.make(env_key, render_mode=render_mode)
    env.reset(seed=seed)
    return env

# Create the environment using make_env
env = make_env('MiniGrid-DoorKey-8x8-v0', seed=69, render_mode='rgb_array')

# Set up the plot to update
plt.ion()  # Turn on interactive mode for real-time updates
fig, ax = plt.subplots()
for i in range(30):
    print(f"Iteration {i + 1}")

    # Sample a random action
    action = env.action_space.sample()

    # Take a step in the environment
    obs, reward, done, truncated, info = env.step(action)

    # Render the environment frame as an RGB array
    frame = env.render()


    # Display the frame using matplotlib
    ax.clear()  # Clear the previous frame
    ax.imshow(frame)
    ax.axis('off')
    ax.set_title(f"Iteration {i + 1}")  # Title with iteration number

    # Pause to update the plot
    plt.draw()
    plt.pause(0.1)  # Pause for 100ms before updating

# After the loop is done, print the final message and close the environment
print("Loop completed, closing the environment and plot.")
env.close()

# Turn off interactive mode and close the plot
plt.ioff()
plt.close()