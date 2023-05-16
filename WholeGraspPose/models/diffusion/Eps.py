import torch
from torch import nn

from WholeGraspPose.models.diffusion.UNet1D import UNet1D
from WholeGraspPose.models.diffusion.utils import get_timestep_embedding


class Eps(nn.Module):
    def __init__(self, D=512, *args, **kwargs):
        super(Eps, self).__init__(*args, **kwargs)

        self.time_emb_dim = D
        self.model = UNet1D()

    def forward(self, x, t, condition):
        t_emb = get_timestep_embedding(timesteps=t, embedding_dim=self.time_emb_dim)
        _x = torch.cat(((x + t_emb), condition), dim=1)

        return self.model(feature_vec=x, time=t_emb, condition=condition)
