import argparse
import numpy as np
import torch
import torch.nn.functional as F
import utils
import minigrid
from utils import device
from model import EarlyFusionTransformerACModel

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
                    help="pause duration between actions (default: 0.1)")
parser.add_argument("--gif", type=str, default=None,
                    help="store output as gif with the given filename")
parser.add_argument("--episodes", type=int, default=1000000,
                    help="number of episodes to visualize")
# Additional parameters for the model.
parser.add_argument("--patch-size", type=int, default=4,
                    help="patch size (default: 4)")
parser.add_argument("--embed-dim", type=int, default=64,
                    help="embedding dimension for the transformer (default: 64)")
parser.add_argument("--num-heads", type=int, default=2,
                    help="number of attention heads (default: 2)")
parser.add_argument("--num-layers", type=int, default=2,
                    help="number of transformer encoder layers (default: 2)")
parser.add_argument("--use-text", action="store_true", default=False,
                    help="whether to use text modality")
parser.add_argument("--vocab-size", type=int, default=50,
                    help="vocabulary size for text instructions (default: 50)")
parser.add_argument("--text-embed-dim", type=int, default=32,
                    help="embedding dimension for text tokens (default: 32)")
parser.add_argument("--max-text-len", type=int, default=20,
                    help="maximum text token length (default: 20)")

args = parser.parse_args()

# Set the random seed.
utils.seed(args.seed)
print(f"Device: {device}\n")

# Create the environment.
env = utils.make_env(args.env, args.seed, render_mode="human")
for _ in range(args.shift):
    env.reset()
print("Environment loaded\n")

# Load observation preprocessor.
obs_space, preprocess_obss = utils.get_obss_preprocessor(env.observation_space)
print("Observation preprocessor loaded")

# Load the trained model.
model_dir = utils.get_model_dir(args.model)
acmodel = EarlyFusionTransformerACModel(
    obs_space, 
    env.action_space,  # Or use envs[0].action_space if you have multiple processes.
    patch_size=args.patch_size,
    num_layers=args.num_layers,
    num_heads=args.num_heads,
    embed_dim=args.embed_dim,
    use_text=args.use_text,
    vocab_size=args.vocab_size,
    text_embed_dim=args.text_embed_dim,
    max_text_len=args.max_text_len
)
state = utils.get_model_state(model_dir)
acmodel.load_state_dict(state)
acmodel.to(device)
acmodel.eval()
print("Model loaded:")
print(acmodel)

# Define agent wrapper with an updated get_action.
class EarlyFusionTransformerAgent:
    def __init__(self, acmodel, preprocess_fn, argmax=False, device=device):
        self.acmodel = acmodel
        self.preprocess_fn = preprocess_fn
        self.argmax = argmax
        self.device = device

    def get_action(self, raw_obs):
        # Preprocess the raw observation.
        preprocessed_obs = self.preprocess_fn([raw_obs])
        # Convert the DictList to a plain dict and move tensors to device.
        preprocessed_obs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                            for k, v in preprocessed_obs.items()}
        with torch.no_grad():
            dist, _ = self.acmodel(preprocessed_obs)
        if self.argmax:
            action = torch.argmax(dist.probs, dim=1)
        else:
            action = dist.sample()
        return action.item()

    def analyze_feedback(self, reward, done):
        pass

agent = EarlyFusionTransformerAgent(acmodel, preprocess_obss, argmax=args.argmax, device=device)
print("Agent loaded\n")

if args.gif:
    from array2gif import write_gif
    frames = []

env.render()

# Run the agent in the environment.
for episode in range(args.episodes):
    raw_obs, _ = env.reset()
    while True:
        env.render()
        if args.gif:
            frames.append(np.moveaxis(env.get_frame(), 2, 0))
        action = agent.get_action(raw_obs)
        raw_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        #print(reward, terminated, truncated)
        agent.analyze_feedback(reward, done)
        if done:
            break

if args.gif:
    print("Saving gif... ", end="")
    write_gif(np.array(frames), args.gif + ".gif", fps=1/args.pause)
    print("Done.")
