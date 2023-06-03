import torch
from torch import nn

from WholeGraspPose.models.diffusion.DenoisingModels import UNet1D, FlatPush, TransformerDenoising
from WholeGraspPose.models.diffusion.utils import get_timestep_embedding


class Eps(nn.Module):
    def __init__(self, D=512, *args, **kwargs):
        super(Eps, self).__init__(*args, **kwargs)

        self.time_emb_dim = D
        # self.model = UNet1D(drop_out_p=0)
        # self.model = FlatPush(depth=5, drop_out_p=0, input_dim=D, out_dim=D)
        #
        # self.time_embed = nn.Sequential(
        #     nn.Linear(self.time_emb_dim , self.time_emb_dim * 4),
        #     nn.SiLU(),
        #     nn.Linear(self.time_emb_dim * 4 , self.time_emb_dim),
        # )
        self.model = TransformerDenoising(seq_len=16, vec_dim=D, drop_out_p=0.2, heads=16, depth=8)

    def forward(self, x, t, condition=None):
        t_emb =  get_timestep_embedding(t, self.time_emb_dim)
        # _x = torch.cat(((x + t_emb), condition), dim=1)

        return self.model(feature_vec=x, time=t_emb, condition=condition)
