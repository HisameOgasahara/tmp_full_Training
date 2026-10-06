"""A small causal decoder sharing its backbone across both prediction tasks."""

import math
import torch
from torch import nn
from torch.nn import functional as F


class CausalBlock(nn.Module):
    def __init__(self, width, heads, ff_multiplier):
        super().__init__()
        self.heads = heads
        self.norm_attention = nn.LayerNorm(width)
        self.project_qkv = nn.Linear(width, 3 * width)
        self.project_output = nn.Linear(width, width)
        self.norm_mlp = nn.LayerNorm(width)
        self.mlp = nn.Sequential(nn.Linear(width, ff_multiplier * width), nn.GELU(), nn.Linear(ff_multiplier * width, width))

    def forward(self, x):
        batch, length, width = x.shape
        qkv = self.project_qkv(self.norm_attention(x)).reshape(batch, length, 3, self.heads, width // self.heads)
        q, k, v = qkv.unbind(dim=2)
        attention = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
        x = x + self.project_output(attention.transpose(1, 2).reshape(batch, length, width))
        return x + self.mlp(self.norm_mlp(x))


class MazeTransformer(nn.Module):
    def __init__(self, vocabulary_size, context_length, config):
        super().__init__()
        width = config["width"]
        if width % config["heads"]:
            raise ValueError("width는 heads로 나누어떨어져야 합니다.")
        self.embed_tokens = nn.Embedding(vocabulary_size, width)
        self.embed_positions = nn.Embedding(context_length, width)
        self.blocks = nn.ModuleList([CausalBlock(width, config["heads"], config["ff_multiplier"]) for _ in range(config["layers"])])
        self.norm_final = nn.LayerNorm(width)
        self.output_bias = nn.Parameter(torch.zeros(vocabulary_size))
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, tokens):
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        x = self.embed_tokens(tokens) * math.sqrt(self.embed_tokens.embedding_dim) + self.embed_positions(positions)
        for block in self.blocks:
            x = block(x)
        # Only the query token needs a prediction. Avoid allocating B*T*V logits.
        return F.linear(self.norm_final(x[:, -1]), self.embed_tokens.weight, self.output_bias)
