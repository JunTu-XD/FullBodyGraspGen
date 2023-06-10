import torch
from torch import nn

from WholeGraspPose.models.diffusion.DenoisingModels import UNet1D, FlatPush, TransformerDenoising
from WholeGraspPose.models.diffusion.utils import get_timestep_embedding


class Eps(nn.Module):
    def __init__(self, D = 512, *args, **kwargs):
        super(Eps, self).__init__(*args, **kwargs)

        self.time_emb_dim = D
        # self.model = UNet1D(drop_out_p=0)
        self.model = FlatPush(depth=3, drop_out_p=0, input_dim=16, out_dim=16)
        self.time_embed = nn.Sequential(
            nn.Linear(self.time_emb_dim , self.time_emb_dim * 2),
            nn.SiLU(),
            nn.Linear(self.time_emb_dim * 2 , self.time_emb_dim),
        )
        # self.cond_mapping = nn.Sequential(
        #     nn.Linear(2, int(D / 2)),
        #     nn.SiLU(),
        #     nn.Linear(int(D / 2), D),
        # )
        # self.model = TransformerDenoising(seq_len=4, vec_dim=D, drop_out_p=0.2, heads=4, depth=3)

    def forward(self, x, t, condition):
        t_emb = self.time_embed(get_timestep_embedding(t, self.time_emb_dim))
        # _condition = self.cond_mapping(condition)

        return self.model(feature_vec=x, time=t_emb, condition= None)