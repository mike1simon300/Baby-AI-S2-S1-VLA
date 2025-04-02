import argparse
import numpy as np
import torch
import torch.nn.functional as F
from model import SmallTransformerACModel
import utils
import minigrid
from utils import device

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--env", required=True,
                    help="name of the environment (REQUIRED)")
parser.add_argument("--model", required=True,
                    help="name of the trained model (REQUIRED)")
parser.add_argument("--seed", type=int, default=0,
                    help="random seed (default: 0)")
parser.add_argument("--shift", type=int, default=0,
                    help="number of times the environment is reset at the beginning (default: 0)")
parser.add_argument("--argmax", action="store_true", default=False,
                    help="select the action with highest probability (default: False)")
parser.add_argument("--pause", type=float, default=0.1,
                    help="pause duration between two consequent actions of the agent (default: 0.1)")
parser.add_argument("--gif", type=str, default=None,
                    help="store output as gif with the given filename")
parser.add_argument("--episodes", type=int, default=1000000,
                    help="number of episodes to visualize")
parser.add_argument("--patch-size", type=int, default=2,
                    help="Patch size for the transformer model (default: 2)")
parser.add_argument("--embed-dim", type=int, default=32,
                    help="Embedding dimension for the transformer model (default: 32)")
parser.add_argument("--num-heads", type=int, default=2,
                    help="Number of attention heads (default: 2)")
parser.add_argument("--num-layers", type=int, default=2,
                    help="Number of transformer encoder layers (default: 2)")

args = parser.parse_args()

# Set seed for all randomness sources
utils.seed(args.seed)
print(f"Device: {device}\n")

# Load environment (with render mode "human")
env = utils.make_env(args.env, args.seed, render_mode="human")
for _ in range(args.shift):
    env.reset()
print("Environment loaded\n")

# Load the transformer model from the saved state.
model_dir = utils.get_model_dir(args.model)

acmodel = SmallTransformerACModel(
    env.observation_space, 
    env.action_space, 
    patch_size=args.patch_size, 
    embed_dim=args.embed_dim, 
    num_heads=args.num_heads, 
    num_layers=args.num_layers
)


state = utils.get_model_state(model_dir)  # load the saved state dictionary
acmodel.load_state_dict(state)
acmodel.to(device)
acmodel.eval()  # set model to evaluation mode

# Create a simple agent wrapper for the transformer model.
class TransformerAgent:
    def __init__(self, acmodel, argmax=False, device=device):
        self.acmodel = acmodel
        self.argmax = argmax
        self.device = device

    def get_action(self, obs):
        # Assume obs is a dictionary with key "image" containing a numpy array.
        # If not, adjust accordingly.
        if isinstance(obs, dict):
            image = obs["image"]
        else:
            image = obs
        # Convert to torch tensor (if not already) and add batch dimension.
        if not torch.is_tensor(image):
            image = torch.tensor(image, dtype=torch.float32)
        image = image.unsqueeze(0).to(self.device)
        # Build observation dict as expected by the model.
        obs_tensor = {"image": image}
        with torch.no_grad():
            dist, _ = self.acmodel(obs_tensor)
        if self.argmax:
            action = torch.argmax(dist.probs, dim=1)
        else:
            action = dist.sample()
        return action.item()

    def analyze_feedback(self, reward, done):
        # For visualization we don't need to process feedback.
        pass

agent = TransformerAgent(acmodel, argmax=args.argmax, device=device)
print("Agent loaded\n")

# Optionally, prepare to save a gif
if args.gif:
    from array2gif import write_gif
    frames = []

# Create a window to view the environment
env.render()

# Run the agent in the environment.
for episode in range(args.episodes):
    obs, _ = env.reset()
    while True:
        env.render()
        if args.gif:
            frames.append(np.moveaxis(env.get_frame(), 2, 0))
        action = agent.get_action(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        agent.analyze_feedback(reward, done)
        if done:
            break

# Save gif if requested.
if args.gif:
    print("Saving gif... ", end="")
    write_gif(np.array(frames), args.gif + ".gif", fps=1/args.pause)
    print("Done.")
