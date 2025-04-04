import torch
import torch.nn as nn
import gymnasium as gym
import numpy as np
from VLA2Systems.task_data_generator import TaskDataGenerator
from minigrid.core.actions import Actions
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.env_util import make_vec_env
from sentence_transformers import SentenceTransformer
from gymnasium import spaces
# Example: you could use a pre-trained sentence embedding model for this
from sentence_transformers import SentenceTransformer

# Assuming your custom environment is called `BabyAIEnv`
class CustomEnvWrapper(gym.ObservationWrapper):
    def __init__(self, env, text_encoder, device=None):
        super().__init__(env)
        # Detect device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        # self.text_encoder = text_encoder.to(self.device)
        self.text_encoder = text_encoder
        self.mission = ""
        # Define a new observation space based on the custom space
        # Assuming the mission space needs to be converted into a vector
        # This is just a placeholder; adapt according to your custom space's structure
        self.mission_space = self.env.observation_space.spaces['mission']  # Example
        
        # Example: If the mission space is a string, we might convert it to an embedding or integer
        self.mission_embedding_dim = 384  # example size for your embedding
        
        # Modify observation space (e.g., adding embedding space)
        self.observation_space = spaces.Dict({
            'image': spaces.Box(low=0, high=1, shape=(7, 7, 8), dtype=np.uint8),
            'direction': self.env.observation_space.spaces['direction'],
            'mission': spaces.Box(low=-np.inf, high=np.inf, shape=(self.mission_embedding_dim,), dtype=np.float32)  # mission embedding
        })
        
    def observation(self, obs):
        # Preprocess the mission space
        # Convert mission string (or whatever type it is) into an embedding or vector
        self.mission = obs['mission']
        mission = obs['mission']  # example: mission might be a string or other format
        # Convert to embedding (this can be done using a pre-trained model or simple encoding)
        # mission_embedding = torch.tensor(self.text_encoder.encode(mission), dtype=torch.float32).to(self.device)
        mission_embedding = self.text_encoder.encode(mission)

        # Process the image and ensure it's also on the same device
        img_enc = obs2enc(obs)  
        # img_enc = torch.tensor(img_enc, dtype=torch.float32).to(self.device)
        obs["image"] = img_enc
        obs["mission"] = mission_embedding

        return obs
    
def obs2enc(obs):
    # Initialize an empty array for the encoded image (expected output shape)
    enc_image = np.zeros((8, 7, 7), dtype=np.float16)  # Shape: channels x height x width
    
    # Get the image from the observation (assuming the shape is [7, 7, 8])
    im = obs["image"].astype(np.int8)  # Ensure it's an integer type for modulo operations
    
    # Transpose dimensions (7, 7, 8) -> (8, 7, 7)
    im = np.transpose(im, (2, 0, 1))  # Rearranging from [height, width, channels] -> [channels, height, width]
    
    # Split the channels into binary values as you were doing
    enc_image[0] = im[0] % 2
    enc_image[1] = im[0] // 2 % 2
    enc_image[2] = im[0] // 4 % 2
    enc_image[3] = im[1] % 2
    enc_image[4] = im[1] // 2 % 2
    enc_image[5] = im[1] // 4 % 2
    enc_image[6] = im[2] % 2
    enc_image[7] = im[2] // 2 % 2
    
    # Final transposition to get the shape as (height, width, channels) [7, 7, 8]
    enc_image = np.transpose(enc_image, (1, 2, 0))  # [7, 7, 8]
    enc_image = enc_image.astype(np.int8)  # Ensure it stays in int8
    return enc_image

def obs2enc2(obs):
    enc_image = np.zeros((8, 7, 7), dtype=np.float16)
    im = obs["image"].astype(np.int8)
    im = np.permute_dims(im, (2, 0, 1))
    enc_image[0] = im[0]%2
    enc_image[1] = im[0]/2%2
    enc_image[2] = im[0]/4%2
    enc_image[3] = im[1]%2
    enc_image[4] = im[1]/2%2
    enc_image[5] = im[1]/4%2
    enc_image[6] = im[2]%2
    enc_image[7] = im[2]/2%2
    enc_image = np.permute_dims(enc_image, (1, 2, 0))
    enc_image = enc_image.astype(np.int8)
    return enc_image

# Define Environment Function for Parallelism
def make_env(generator: TaskDataGenerator, difficulty, text_encoder):
    def _make():
        generator.reset(difficulty=difficulty)
        return CustomEnvWrapper(generator.env, text_encoder)
    return _make

# Custom Feature Extractor using your Transformer Model
class SimpleVLA(BaseFeaturesExtractor):
    """
    Custom feature extractor that takes a dictionary observation with keys:
      - 'image': grid map data, shape (7, 7, 8)
      - 'mission': pre-embedded mission vector, shape (sentence_embedding_size,)
    and returns a latent vector for the policy.
    """
    def __init__(self, observation_space: gym.spaces.Dict, embedding_size=128, sentence_embedding_size=384):
        # Compute the size of the latent vector output.
        # Here we flatten the transformer output which is expected to be (batch, 50, embedding_size)
        latent_dim = 50 * embedding_size
        super(SimpleVLA, self).__init__(observation_space, features_dim=latent_dim)
        self.embedding_size = embedding_size
        # Define projectors
        self.grid_projector = nn.Linear(8, embedding_size)
        self.text_projector = nn.Linear(sentence_embedding_size, embedding_size)

        # Transformer encoder: we assume 3 layers and 8 heads (you can adjust as needed)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_size, nhead=8)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)

    def forward(self, observations: dict) -> torch.Tensor:
        # Expecting observations to be a dict with keys "image" and "mission"
        # Convert inputs to float tensors
        # observations["image"]: shape (batch, 7, 7, 8)
        # observations["mission"]: shape (batch, sentence_embedding_size)
        grid_map = observations["image"].float()  # shape: (batch, 7, 7, 8)
        mission = observations["mission"].float()   # shape: (batch, sentence_embedding_size)
        batch_size = grid_map.shape[0]

        # Project grid map: result shape -> (batch, 7, 7, embedding_size)
        grid_emb = self.grid_projector(grid_map)
        # Flatten grid to tokens: (batch, 49, embedding_size)
        grid_emb = grid_emb.view(batch_size, 49, self.embedding_size)

        # Project mission text: (batch, embedding_size)
        text_emb = self.text_projector(mission)
        # Add a token dimension: (batch, 1, embedding_size)
        text_emb = text_emb.unsqueeze(1)

        # Concatenate text token to grid tokens: (batch, 50, embedding_size)
        tokens = torch.cat((grid_emb, text_emb), dim=1)

        # Transformer expects input shape (sequence_length, batch, embedding_size)
        tokens = tokens.transpose(0, 1)  # now (50, batch, embedding_size)
        transformed = self.transformer(tokens)  # output shape: (50, batch, embedding_size)
        transformed = transformed.transpose(0, 1)  # (batch, 50, embedding_size)

        # Flatten the tokens to form a feature vector: (batch, 50 * embedding_size)
        latent = transformed.flatten(start_dim=1)
        return latent


# Create and train the PPO agent with our custom policy
def train_agent():
    # Custom policy can use the standard MLP head on top of our features extractor.
    policy_kwargs = dict(
        features_extractor_class=SimpleVLA,
        features_extractor_kwargs=dict(embedding_size=128, sentence_embedding_size=384),
        net_arch=[dict(pi=[256, 256], vf=[256, 256])]  # adjust network architecture as needed
    )

    # Environment setup (adjust number of envs as needed)
    env_dict = {
        "easy": ["BabyAI-ActionObjDoor-v0"],
        "intermediate": ["BabyAI-FindObjS5-v0"],
        "hard": ["BabyAI-UnlockToUnlock-v0", "BabyAI-Synth-v0"]
    }
    
    difficulty = "easy"
    n_envs = 32  # number of parallel environments
    # n_envs = 8  # number of parallel environments
    generator = TaskDataGenerator(env_dict)
    text_encoder = SentenceTransformer("all-MiniLM-L6-v2")

    envs = SubprocVecEnv([make_env(generator, difficulty, text_encoder) for _ in range(n_envs)])

    model = PPO("MultiInputPolicy", envs, verbose=1, learning_rate=3e-4,
                n_steps=2048, batch_size=64, n_epochs=10, gamma=0.99, clip_range=0.2,
                policy_kwargs=policy_kwargs)
    model.learn(total_timesteps=1000_000)
    model.save("ppo_transformer_agent")

if __name__ == "__main__":
    train_agent()
