import math 
import random 
from dataclasses import dataclass 
from typing import List, Tuple, Optional

import torch 
import torch.nn as nn 
import torch.nn.functional as F

# images have shape (B, C, H, W)
# B = batch size, C = channels, H = height, W = width
# patch tokens have shape (B, N, D) where N = number of patches, D = embedding dimension 


class PatchEmbed(nn.Module):
    """converts the image into a sequence of patch embeddings
    Input = (B , C, H, W)
    Ouput = (B, N, D) where N = (H/P) * (W/P)
    """

    def __init__(
            self, 
            img_size: int = 224, 
            patch_size: int = 16, 
            in_chans: int = 3, 
            embed_dim: int = 768, # the dimension of the embedding that represents each patch 
    ):
        super().__init__()
        assert img_size % patch_size == 0, "img size must be dividable by the patch size"
        self.img_size = img_size 
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size

        # this is a learned module 
        self.proj = nn.Conv2d(
            in_channels = in_chans, 
            out_channels = embed_dim, 
            stride = patch_size, 
            kernel_size = patch_size,
            bias = True,
        )
    
    # how the data flows through the module when you call y = patch_embed(x)
    # apply proj and then reshape into tokens   
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x) # (B,D,G,G) 
        x = x.flatten(2).transpose(1, 2) # (B, N, D)
        return x

class MultiHeadSelfAttention(nn.Module):
    def __init__(
            self, 
            dim: int, 
            num_heads: int, 
            attn_drop: float = 0.0, 
            proj_drop: float = 0.0, 
    ):
        super().__init__()
        assert dim % num_heads == 0, "the dim must be divisible by the number of heads"

        self.dim = dim 
        self.num_heads = num_heads 
        self.head_dim = dim // num_heads 
        # scaling factor for the dot product attention 
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, 3*dim, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass



