import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import torch_ac


# Function from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def init_params(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)


class ACModel(nn.Module, torch_ac.RecurrentACModel):
    def __init__(self, obs_space, action_space, use_memory=False, use_text=False):
        super().__init__()

        # Decide which components are enabled
        self.use_text = use_text
        self.use_memory = use_memory

        # Define image embedding
        self.image_conv = nn.Sequential(
            nn.Conv2d(3, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU()
        )
        n = obs_space["image"][0]
        m = obs_space["image"][1]
        self.image_embedding_size = ((n-1)//2-2)*((m-1)//2-2)*64

        # Define memory
        if self.use_memory:
            self.memory_rnn = nn.LSTMCell(self.image_embedding_size, self.semi_memory_size)

        # Define text embedding
        if self.use_text:
            self.word_embedding_size = 32
            self.word_embedding = nn.Embedding(obs_space["text"], self.word_embedding_size)
            self.text_embedding_size = 128
            self.text_rnn = nn.GRU(self.word_embedding_size, self.text_embedding_size, batch_first=True)

        # Resize image embedding
        self.embedding_size = self.semi_memory_size
        if self.use_text:
            self.embedding_size += self.text_embedding_size

        # Define actor's model
        self.actor = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, action_space.n)
        )

        # Define critic's model
        self.critic = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # Initialize parameters correctly
        self.apply(init_params)

    @property
    def memory_size(self):
        return 2*self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.image_embedding_size

    def forward(self, obs, memory):
        x = obs.image.transpose(1, 3).transpose(2, 3)
        x = self.image_conv(x)
        x = x.reshape(x.shape[0], -1)

        if self.use_memory:
            hidden = (memory[:, :self.semi_memory_size], memory[:, self.semi_memory_size:])
            hidden = self.memory_rnn(x, hidden)
            embedding = hidden[0]
            memory = torch.cat(hidden, dim=1)
        else:
            embedding = x

        if self.use_text:
            embed_text = self._get_embed_text(obs.text)
            embedding = torch.cat((embedding, embed_text), dim=1)

        x = self.actor(embedding)
        dist = Categorical(logits=F.log_softmax(x, dim=1))

        x = self.critic(embedding)
        value = x.squeeze(1)

        return dist, value, memory

    def _get_embed_text(self, text):
        _, hidden = self.text_rnn(self.word_embedding(text))
        return hidden[-1]
    
class ACModel(nn.Module):
    def __init__(self, obs_space, action_space, use_memory=False, use_text=False, max_text_len=32):
        super().__init__()

        self.use_text = use_text
        self.use_memory = use_memory
        self.max_text_len = max_text_len

        # ---- Image embedding ----
        self.image_conv = nn.Sequential(
            nn.Conv2d(3, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU()
        )
        n, m = obs_space["image"][0], obs_space["image"][1]
        self.image_embedding_size = ((n - 1) // 2 - 2) * ((m - 1) // 2 - 2) * 64

        # ---- Memory ----
        if self.use_memory:
            self.memory_rnn = nn.LSTMCell(self.image_embedding_size, self.semi_memory_size)

        # ---- Text embedding via Transformer ----
        if self.use_text:
            self.word_embedding_size = 32
            self.text_embedding_size = 128
            self.vocab_size = obs_space["text"]
            self.word_embedding = nn.Embedding(self.vocab_size, self.word_embedding_size)

            self.pos_embedding = nn.Parameter(torch.randn(1, max_text_len, self.word_embedding_size))
            encoder_layer = TransformerEncoderLayer(
                d_model=self.word_embedding_size,
                nhead=4,
                dim_feedforward=256,
                batch_first=True
            )
            self.text_encoder = TransformerEncoder(encoder_layer, num_layers=2)

        # ---- Embedding combination ----
        self.embedding_size = self.semi_memory_size
        if self.use_text:
            self.embedding_size += self.text_embedding_size

        # ---- Actor ----
        self.actor = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, action_space.n)
        )

        # ---- Critic ----
        self.critic = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        self.apply(self.init_params)

    @property
    def memory_size(self):
        return 2 * self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.image_embedding_size

    def forward(self, obs, memory):
        x = obs.image.transpose(1, 3).transpose(2, 3)  # NHWC → NCHW
        x = self.image_conv(x)
        x = x.reshape(x.size(0), -1)

        if self.use_memory:
            h, c = memory[:, :self.semi_memory_size], memory[:, self.semi_memory_size:]
            h, c = self.memory_rnn(x, (h, c))
            embedding = h
            memory = torch.cat([h, c], dim=1)
        else:
            embedding = x

        if self.use_text:
            text_embedding = self._get_embed_text(obs.text)
            embedding = torch.cat([embedding, text_embedding], dim=1)

        logits = self.actor(embedding)
        dist = Categorical(logits=F.log_softmax(logits, dim=1))

        value = self.critic(embedding).squeeze(1)

        return dist, value, memory

    def _get_embed_text(self, text):
        # text: (batch, seq_len)
        x = self.word_embedding(text)  # (B, T, E)
        seq_len = x.size(1)
        pos_embed = self.pos_embedding[:, :seq_len, :]
        x = x + pos_embed

        encoded = self.text_encoder(x)  # (B, T, E)

        # Mean pooling over sequence
        return encoded.mean(dim=1)  # (B, E)

    def init_params(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
class SmallTransformerACModel(nn.Module):
    def __init__(self, obs_space, action_space, patch_size=4, num_layers=2, num_heads=2, embed_dim=8):
        super().__init__()
        # Correctly assign dimensions by extracting the shape from the Box
        image_shape = obs_space["image"].shape if hasattr(obs_space["image"], "shape") else obs_space["image"]
        self.img_width, self.img_height, self.num_channels = image_shape
        
        self.patch_size = patch_size
        self.num_patches = (self.img_height // patch_size) * (self.img_width // patch_size)
        
        # Linear projection of flattened patches
        self.patch_embed = nn.Linear(self.num_channels * patch_size * patch_size, embed_dim)
        
        # Positional encoding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Final aggregation: simple average pooling
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # Actor and critic heads
        self.actor = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.Tanh(),
            nn.Linear(64, action_space.n)
        )
        self.critic = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        self.init_weights()
    
    def init_weights(self):
        nn.init.normal_(self.pos_embed, std=0.02)
    
    def forward(self, obs, memory=None):
        # Try to get the image via attribute; if that fails, use key access.
        try:
            x = obs.image
        except AttributeError:
            x = obs["image"]

        # x is expected to be in [B, W, H, C] order (babyAI)
        # If the second dimension is not equal to the number of channels, permute:
        if x.shape[1] != self.num_channels:
            x = x.permute(0, 3, 2, 1)  # now [B, C, H, W]
        
        B, C, H, W = x.shape

        # Extract patches along H and W dimensions.
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)

        # patches shape: [B, C, num_patches_h, num_patches_w, patch_size, patch_size]
        patches = patches.contiguous().view(B, C, -1, self.patch_size, self.patch_size)
        # Rearrange so that patches are in the second dimension:
        patches = patches.permute(0, 2, 1, 3, 4)  # [B, num_patches, C, patch_size, patch_size]
        patches = patches.contiguous().view(B, self.num_patches, -1)  # flatten each patch

        # Compute patch embeddings.
        patch_embeddings = self.patch_embed(patches)  # [B, num_patches, embed_dim]
        x = patch_embeddings + self.pos_embed          # [B, num_patches, embed_dim]

        # Transformer expects input shape [sequence, batch, embed_dim]
        x = x.transpose(0, 1)  # [num_patches, B, embed_dim]
        x = self.transformer_encoder(x)
        x = x.transpose(0, 1)  # [B, num_patches, embed_dim]

        # Aggregate patch embeddings (mean pooling)
        x = x.mean(dim=1)      # [B, embed_dim]

        # Actor and critic heads.
        logits = self.actor(x)
        dist = Categorical(logits=F.log_softmax(logits, dim=1))
        value = self.critic(x).squeeze(1)

        return dist, value

    @property
    def recurrent(self):
        return False
    
class EarlyFusionTransformerACModel(nn.Module):
    def __init__(self, obs_space, action_space, patch_size=4, num_layers=2, num_heads=2,
                 embed_dim=64, use_text=False, vocab_size=None, text_embed_dim=32, max_text_len=20):
        """
        Args:
            obs_space: a dict-like object with observation info. Expects obs_space["image"] to be a tuple (W, H, C)
                       or a Box with that shape.
            action_space: an object with attribute 'n' (number of actions).
            patch_size: size of each square patch from the image.
            num_layers: number of transformer encoder layers.
            num_heads: number of attention heads.
            embed_dim: common embedding dimension.
            use_text: whether to use text modality.
            vocab_size: vocabulary size (required if use_text is True).
            text_embed_dim: embedding dimension for text tokens.
            max_text_len: maximum text token length.
        """
        super().__init__()
        # ------------- Image Branch -------------
        # Determine image shape from obs_space["image"]
        image_shape = obs_space["image"].shape if hasattr(obs_space["image"], "shape") else obs_space["image"]
        self.img_width, self.img_height, self.num_channels = image_shape

        self.patch_size = patch_size
        # Compute number of patches based on image dimensions and patch size.
        self.num_patches = (self.img_height // patch_size) * (self.img_width // patch_size)
        self.patch_embed = nn.Linear(self.num_channels * patch_size * patch_size, embed_dim)

        # ------------- Text Branch -------------
        self.use_text = use_text
        self.max_text_len = max_text_len
        if self.use_text:
            if vocab_size is None:
                raise ValueError("vocab_size must be provided if use_text is True")
            self.word_embedding = nn.Embedding(vocab_size, text_embed_dim)
            self.text_proj = nn.Linear(text_embed_dim, embed_dim)

        # ------------- Joint Token Preparation -------------
        # Learnable CLS token (for pooling)
        self.cls_token = nn.Parameter(torch.zeros(1, embed_dim))
        # We initially set a maximum sequence length based on an expected length:
        # 1 (CLS) + max_text_len (if text is used) + self.num_patches.
        self.max_seq_len = 1 + (max_text_len if self.use_text else 0) + self.num_patches
        # Learnable positional embeddings of shape (1, max_seq_len, embed_dim).
        self.pos_embed = nn.Parameter(torch.zeros(1, self.max_seq_len, embed_dim))
        # Modality embeddings: one for text and one for image.
        if self.use_text:
            self.text_mod_embed = nn.Parameter(torch.zeros(1, embed_dim))
        self.img_mod_embed = nn.Parameter(torch.zeros(1, embed_dim))

        # ------------- Transformer Encoder -------------
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # ------------- Actor and Critic Heads -------------
        self.actor = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.Tanh(),
            nn.Linear(64, action_space.n)
        )
        self.critic = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)
        if self.use_text:
            nn.init.normal_(self.text_mod_embed, std=0.02)
        nn.init.normal_(self.img_mod_embed, std=0.02)
        # You might also initialize patch_embed and text_proj if desired.

    def forward(self, obs, memory=None):
        # ------------- Process Image -------------
        try:
            x_img = obs.image
        except AttributeError:
            x_img = obs["image"]
        # Sometimes, obs["image"] is a DictList; convert it to a tensor if needed.
        if not isinstance(x_img, torch.Tensor):
            x_img = torch.stack(list(x_img))
        B = x_img.shape[0]
        # Expect image in [B, W, H, C]. If not, permute.
        if x_img.shape[1] != self.num_channels:
            x_img = x_img.permute(0, 3, 2, 1)  # becomes [B, C, H, W]
        # Extract patches:
        patches = x_img.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        # patches: [B, C, num_patches_h, num_patches_w, patch_size, patch_size]
        patches = patches.contiguous().view(B, self.num_channels, -1, self.patch_size, self.patch_size)
        patches = patches.permute(0, 2, 1, 3, 4)  # [B, num_patches, C, patch_size, patch_size]
        patches = patches.contiguous().view(B, self.num_patches, -1)  # [B, num_patches, C*patch_size*patch_size]
        img_tokens = self.patch_embed(patches)  # [B, num_patches, embed_dim]
        img_tokens = img_tokens + self.img_mod_embed  # add modality embedding to image tokens

        # ------------- Process Text -------------
        if self.use_text:
            try:
                x_text = obs.text
            except AttributeError:
                x_text = obs["text"]
            # x_text should be of shape [B, seq_len], with seq_len <= max_text_len.
            text_embeds = self.word_embedding(x_text)  # [B, seq_len, text_embed_dim]
            text_tokens = self.text_proj(text_embeds)    # [B, seq_len, embed_dim]
            text_tokens = text_tokens + self.text_mod_embed  # add modality embedding
        else:
            text_tokens = None

        # ------------- Construct Joint Sequence -------------
        # CLS token:
        cls_tokens = self.cls_token.expand(B, 1, -1)  # [B, 1, embed_dim]
        if self.use_text:
            joint_tokens = torch.cat([cls_tokens, text_tokens, img_tokens], dim=1)
        else:
            joint_tokens = torch.cat([cls_tokens, img_tokens], dim=1)
        L = joint_tokens.shape[1]  # actual joint sequence length
        
        # Interpolate positional embeddings if necessary:
        if L != self.pos_embed.shape[1]:
            pos_embed = F.interpolate(self.pos_embed.transpose(1,2), size=L, mode="linear", align_corners=False).transpose(1,2)
        else:
            pos_embed = self.pos_embed
        
        joint_tokens = joint_tokens + pos_embed

        # ------------- Transformer Encoding -------------
        # Transformer expects input of shape [sequence, batch, embed_dim]
        x = joint_tokens.transpose(0, 1)  # [L, B, embed_dim]
        x = self.transformer_encoder(x)
        x = x.transpose(0, 1)  # [B, L, embed_dim]
        # Use the output of the CLS token (first token)
        rep = x[:, 0, :]  # [B, embed_dim]

        # ------------- Actor and Critic -------------
        logits = self.actor(rep)
        dist = Categorical(logits=F.log_softmax(logits, dim=1))
        value = self.critic(rep).squeeze(1)
        return dist, value

    @property
    def recurrent(self):
        return False
    
class LateFusionTransformerACModel(nn.Module):
    def __init__(self, obs_space, action_space, patch_size=4, num_layers=2, num_heads=2,
                 embed_dim=64, use_text=False, vocab_size=None, text_embed_dim=32, max_text_len=20):
        """
        Args:
            obs_space: dict-like, with obs_space["image"] being a tuple (W, H, C) or a Box with that shape.
            action_space: object with attribute 'n' (number of actions).
            patch_size: size of each square patch from the image.
            num_layers: number of transformer encoder layers (for both branches).
            num_heads: number of attention heads.
            embed_dim: common embedding dimension.
            use_text: whether to use the text modality.
            vocab_size: vocabulary size (required if use_text is True).
            text_embed_dim: embedding dimension for text tokens.
            max_text_len: maximum text token length.
        """
        super().__init__()
        # ------------- Image Branch -------------
        image_shape = obs_space["image"].shape if hasattr(obs_space["image"], "shape") else obs_space["image"]
        self.img_width, self.img_height, self.num_channels = image_shape
        self.patch_size = patch_size
        self.num_patches = (self.img_height // patch_size) * (self.img_width // patch_size)
        
        # Linear projection of flattened patches.
        self.patch_embed = nn.Linear(self.num_channels * patch_size * patch_size, embed_dim)
        # Learnable CLS token for image branch.
        self.img_cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # Positional embeddings for image branch (for CLS + patches).
        self.img_pos_embed = nn.Parameter(torch.zeros(1, 1 + self.num_patches, embed_dim))
        # Transformer encoder for image tokens.
        encoder_layer_img = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.img_transformer_encoder = nn.TransformerEncoder(encoder_layer_img, num_layers=num_layers)
        
        # ------------- Text Branch -------------
        self.use_text = use_text
        if self.use_text:
            if vocab_size is None:
                raise ValueError("vocab_size must be provided if use_text is True")
            # Embedding layer for words.
            self.word_embedding = nn.Embedding(vocab_size, text_embed_dim)
            # Project word embeddings to common embed_dim.
            self.text_proj = nn.Linear(text_embed_dim, embed_dim)
            # Learnable CLS token for text branch.
            self.text_cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            # Positional embeddings for text branch (for CLS + text tokens).
            self.text_pos_embed = nn.Parameter(torch.zeros(1, 1 + max_text_len, embed_dim))
            # Transformer encoder for text tokens.
            encoder_layer_text = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
            self.text_transformer_encoder = nn.TransformerEncoder(encoder_layer_text, num_layers=num_layers)
        
        # ------------- Late Fusion -------------
        # If both modalities are used, we fuse by concatenating their representations.
        fusion_input_dim = embed_dim * 2 if self.use_text else embed_dim
        self.fusion_fc = nn.Linear(fusion_input_dim, embed_dim)
        
        # ------------- Actor and Critic Heads -------------
        self.actor = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.Tanh(),
            nn.Linear(64, action_space.n)
        )
        self.critic = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.img_pos_embed, std=0.02)
        nn.init.normal_(self.img_cls_token, std=0.02)
        if self.use_text:
            nn.init.normal_(self.text_pos_embed, std=0.02)
            nn.init.normal_(self.text_cls_token, std=0.02)
        # Optionally, you could initialize patch_embed and text_proj here as well.

    def forward(self, obs, memory=None):
        # ------------- Process Image Branch -------------
        try:
            x_img = obs.image
        except AttributeError:
            x_img = obs["image"]
        # Ensure x_img is a tensor. Expected shape: [B, W, H, C]. If needed, permute.
        if not isinstance(x_img, torch.Tensor):
            x_img = torch.stack(list(x_img))
        B = x_img.shape[0]
        if x_img.shape[1] != self.num_channels:
            x_img = x_img.permute(0, 3, 2, 1)  # Now shape: [B, C, H, W]
        
        # Extract patches.
        patches = x_img.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        # patches shape: [B, C, num_patches_h, num_patches_w, patch_size, patch_size]
        patches = patches.contiguous().view(B, self.num_channels, -1, self.patch_size, self.patch_size)
        patches = patches.permute(0, 2, 1, 3, 4)  # [B, num_patches, C, patch_size, patch_size]
        patches = patches.contiguous().view(B, self.num_patches, -1)  # [B, num_patches, C*patch_size*patch_size]
        img_tokens = self.patch_embed(patches)  # [B, num_patches, embed_dim]
        # Prepend the image CLS token.
        img_cls_tokens = self.img_cls_token.expand(B, 1, -1)
        img_tokens = torch.cat([img_cls_tokens, img_tokens], dim=1)  # [B, 1+num_patches, embed_dim]
        # Add positional embeddings.
        img_tokens = img_tokens + self.img_pos_embed
        # Transformer expects input shape [sequence, batch, embed_dim].
        img_tokens = img_tokens.transpose(0, 1)
        img_tokens = self.img_transformer_encoder(img_tokens)
        img_tokens = img_tokens.transpose(0, 1)
        # Use the output corresponding to the CLS token.
        img_rep = img_tokens[:, 0, :]  # [B, embed_dim]
        
        # ------------- Process Text Branch -------------
        if self.use_text:
            try:
                x_text = obs.text
            except AttributeError:
                x_text = obs["text"]
            # x_text is expected to be of shape [B, seq_len] (with seq_len <= max_text_len).
            text_embeds = self.word_embedding(x_text)          # [B, seq_len, text_embed_dim]
            text_tokens = self.text_proj(text_embeds)            # [B, seq_len, embed_dim]
            # Prepend the text CLS token.
            text_cls_tokens = self.text_cls_token.expand(B, 1, -1)
            text_tokens = torch.cat([text_cls_tokens, text_tokens], dim=1)  # [B, 1+seq_len, embed_dim]
            L_text = text_tokens.shape[1]
            # Interpolate positional embeddings if necessary.
            if L_text != self.text_pos_embed.shape[1]:
                text_pos_embed = F.interpolate(self.text_pos_embed.transpose(1,2), size=L_text, mode="linear", align_corners=False).transpose(1,2)
            else:
                text_pos_embed = self.text_pos_embed
            text_tokens = text_tokens + text_pos_embed
            text_tokens = text_tokens.transpose(0, 1)  # [sequence, B, embed_dim]
            text_tokens = self.text_transformer_encoder(text_tokens)
            text_tokens = text_tokens.transpose(0, 1)    # [B, 1+seq_len, embed_dim]
            text_rep = text_tokens[:, 0, :]  # [B, embed_dim]
        
        # ------------- Late Fusion -------------
        if self.use_text:
            fused = torch.cat([img_rep, text_rep], dim=1)  # [B, 2*embed_dim]
        else:
            fused = img_rep  # [B, embed_dim]
        fused = self.fusion_fc(fused)  # [B, embed_dim]
        
        # ------------- Actor and Critic Heads -------------
        logits = self.actor(fused)
        dist = Categorical(logits=F.log_softmax(logits, dim=1))
        value = self.critic(fused).squeeze(1)
        return dist, value
    
    @property
    def recurrent(self):
        return False