import torch
from torch import nn

from WholeGraspPose.models.diffusion.DenoisingModels import UNet1D, FlatPush, TransformerDenoising, NNet1D
from WholeGraspPose.models.diffusion.utils import get_timestep_embedding


class Eps(nn.Module):
    def __init__(self, D=512, *args, **kwargs):
        super(Eps, self).__init__(*args, **kwargs)

        self.time_emb_dim = D
        # self.model = NNet1D(drop_out_p=0.1)
        self.model = UNet1D(drop_out_p=0)
        # self.model = FlatPush(depth=5,drop_out_p=0)
        # self.model = TransformerDenoising(vec_dim=128, drop_out_p=0, heads=4, depth=3)

    def forward(self, x, t, condition):
        t_emb = get_timestep_embedding(timesteps=t, embedding_dim=self.time_emb_dim)
        # _x = torch.cat(((x + t_emb), condition), dim=1)

        return self.model(feature_vec=x, time=t_emb, condition=condition)
