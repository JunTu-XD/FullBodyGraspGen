import torch
from torch import nn

from WholeGraspPose.models.diffusion.utils import get_timestep_embedding


class Eps(nn.Module):
    def __init__(self, D, *args, **kwargs):
        super(Eps, self).__init__(*args, **kwargs)

        self.time_emb_dim = D
        self.model = nn.Sequential(*[nn.Linear(D, D)])

    def forward(self, x, t, condition):
        t_emb = get_timestep_embedding(timesteps=t, embedding_dim=self.time_emb_dim)
        _x = torch.cat(((x + t_emb), condition), dim=1)
        assert _x.shape[0] == x.shape[0]
        assert _x.shape[1] == x.shape[1] + self.time_emb_dim
        return self.model(x)
