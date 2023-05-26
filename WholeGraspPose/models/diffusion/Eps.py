import torch
from torch import nn

from WholeGraspPose.models.diffusion.DenoisingModels import UNet1D, FlatPush, TransformerDenoising
from WholeGraspPose.models.diffusion.utils import get_timestep_embedding


class Eps(nn.Module):
    def __init__(self, D=512, *args, **kwargs):
        super(Eps, self).__init__(*args, **kwargs)

        self.time_emb_dim = D
        # self.model = UNet1D(drop_out_p=0)
        self.model = FlatPush(depth=6,drop_out_p=0, input_dim=D, out_dim=D)
        # self.model = TransformerDenoising(vec_dim=128, drop_out_p=0, heads=4, depth=3)

    def forward(self, x, t, condition=None):
        t_emb = get_timestep_embedding(timesteps=t, embedding_dim=self.time_emb_dim)
        if condition is not None:
            _x = torch.cat(((x + t_emb), condition), dim=1)
        else:
            _x = (x + t_emb)

        return self.model(feature_vec=x, time=t_emb, condition=condition)
