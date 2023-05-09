import math

import numpy as np
import torch
from torch import nn
from einops import repeat

def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    """

    :param schedule:
    :param n_timestep:
    :param linear_start:
    :param linear_end:
    :param cosine_s:
    :return:
    beta sequence
    """
    if schedule == "linear":
        betas = (
                torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
        )

    elif schedule == "cosine":
        timesteps = (
                torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * np.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = np.clip(betas, a_min=0, a_max=0.999)

    elif schedule == "sqrt_linear":
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64)
    elif schedule == "sqrt":
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64) ** 0.5
    else:
        raise ValueError(f"schedule '{schedule}' unknown.")
    return betas.numpy()


def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    if not repeat_only:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    else:
        embedding = repeat(timesteps, 'b -> b d', d=dim)
    return embedding

def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)


def Normalize(in_features):
    return torch.nn.BatchNorm1d(eps=1e-6, num_features=in_features, affine=True)

## linear replacing conv in the implementation in DDPM, DDIM
class ResnetBlock(nn.Module):
    def __init__(self, *, in_features, out_features=None, dropout, temb_dim=512):
        super().__init__()

        out_features = in_features if out_features is None else out_features
        self.in_features = in_features
        self.out_features = out_features

        self.norm1 = Normalize(in_features)
        self.norm2 = Normalize(out_features)
        self.dropout = torch.nn.Dropout(dropout)

        self.lin1 = nn.Linear(in_features=in_features, out_features=out_features)
        self.lin2 = nn.Linear(in_features=out_features, out_features=out_features)
        self.align_dim = nn.Linear(in_features=out_features, out_features=in_features)
        self.temb_proj = nn.Linear(in_features=temb_dim, out_features=out_features)

        self.in_features = in_features
        self.out_features = out_features
    def forward(self, x,  temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.lin1(h)

        h = h + self.temb_proj(nonlinearity(temb))

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.lin2(h)

        if self.in_features != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h