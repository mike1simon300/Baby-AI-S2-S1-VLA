import os
import json
import numpy
import re
import torch
import torch_ac
import gymnasium as gym
from sentence_transformers import SentenceTransformer
import utils

def get_obss_preprocessor_sentence(obs_space):
    # Check if obs_space is an image space
    if isinstance(obs_space, gym.spaces.Box):
        obs_space = {"image": obs_space.shape}

        def preprocess_obss(obss, device=None):
            return torch_ac.DictList({
                "image": preprocess_images(obss, device=device)
            })

    # Check if it is a MiniGrid observation space
    elif isinstance(obs_space, gym.spaces.Dict) and "image" in obs_space.spaces.keys():
        obs_space = {"image": obs_space.spaces["image"].shape, "text": 100}

        vocab = Vocabulary(obs_space["text"])
        encoder = SentenceTransformer("all-MiniLM-L6-v2")
        def preprocess_obss(obss, device=None):
            images = preprocess_images([obs["image"] for obs in obss], device=device)
            text_tensor = preprocess_texts([obs["mission"] for obs in obss], vocab, device=device)
            raw_text = numpy.array([obs["mission"] for obs in obss], dtype=object)  # <-- FIX HERE
            with torch.no_grad():
                embbed_text = encoder.encode(raw_text, show_progress_bar=False)
            return torch_ac.DictList({
                "image": images,
                "text": text_tensor,
                "embbed_text": embbed_text,
                "raw_text": raw_text
            })

        preprocess_obss.vocab = vocab
        preprocess_obss.encoder = encoder

    else:
        raise ValueError("Unknown observation space: " + str(obs_space))

    return obs_space, preprocess_obss

def get_obss_preprocessor_sentence(obs_space):
    # Check if obs_space is an image space
    if isinstance(obs_space, gym.spaces.Box):
        obs_space = {"image": obs_space.shape}

        def preprocess_obss(obss, device=None):
            return torch_ac.DictList({
                "image": preprocess_images(obss, device=device)
            })
    # Check if it is a MiniGrid observation space
    elif isinstance(obs_space, gym.spaces.Dict) and "image" in obs_space.spaces.keys():
        obs_space = {"image": obs_space.spaces["image"].shape, "text": 100}

        vocab = Vocabulary(obs_space["text"])
        encoder = SentenceTransformer("all-MiniLM-L6-v2")
        
        def preprocess_obss(obss, device=None):
            images = preprocess_images([obs["image"] for obs in obss], device=device)
            text_tensor = preprocess_texts([obs["mission"] for obs in obss], vocab, device=device)
            # Get raw texts as an array of objects
            raw_text = numpy.array([obs["mission"] for obs in obss], dtype=object)
            
            # Initialize a cache if it doesn't exist
            if not hasattr(preprocess_obss, "cache"):
                preprocess_obss.cache = {}

            cache = preprocess_obss.cache
            embeddings = [None] * len(raw_text)
            new_texts = []
            new_indices = []

            # Look up each mission in the cache
            for i, t in enumerate(raw_text):
                # Use the string value as key (if t is not hashable, convert it to str)
                key = str(t)
                if key in cache:
                    embeddings[i] = cache[key]
                else:
                    new_texts.append(t)
                    new_indices.append(i)

            # If there are texts not in the cache, encode them in a batch
            if new_texts:
                new_emb = encoder.encode(new_texts, show_progress_bar=False)
                for idx, emb in zip(new_indices, new_emb):
                    embeddings[idx] = emb
                    cache[str(raw_text[idx])] = emb

            embbed_text = numpy.array(embeddings)
            return torch_ac.DictList({
                "image": images,
                "text": text_tensor,
                "embbed_text": embbed_text,
                "raw_text": raw_text
            })

        preprocess_obss.vocab = vocab
        preprocess_obss.encoder = encoder

    else:
        raise ValueError("Unknown observation space: " + str(obs_space))

    return obs_space, preprocess_obss



def preprocess_images(images, device=None):
    # Bug of Pytorch: very slow if not first converted to numpy array
    images = numpy.array(images)
    return torch.tensor(images, device=device, dtype=torch.float)


def preprocess_texts(texts, vocab, device=None):
    var_indexed_texts = []
    max_text_len = 0

    for text in texts:
        tokens = re.findall("([a-z]+)", text.lower())
        var_indexed_text = numpy.array([vocab[token] for token in tokens])
        var_indexed_texts.append(var_indexed_text)
        max_text_len = max(len(var_indexed_text), max_text_len)

    indexed_texts = numpy.zeros((len(texts), max_text_len))

    for i, indexed_text in enumerate(var_indexed_texts):
        indexed_texts[i, :len(indexed_text)] = indexed_text

    return torch.tensor(indexed_texts, device=device, dtype=torch.long)


class Vocabulary:
    """A mapping from tokens to ids with a capacity of `max_size` words.
    It can be saved in a `vocab.json` file."""

    def __init__(self, max_size):
        self.max_size = max_size
        self.vocab = {}

    def load_vocab(self, vocab):
        self.vocab = vocab

    def __getitem__(self, token):
        if not token in self.vocab.keys():
            if len(self.vocab) >= self.max_size:
                raise ValueError("Maximum vocabulary capacity reached")
            self.vocab[token] = len(self.vocab) + 1
        return self.vocab[token]
