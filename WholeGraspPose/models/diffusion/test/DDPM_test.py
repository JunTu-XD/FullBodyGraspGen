import torch
from torch import nn

from WholeGraspPose.models.diffusion.DDPM import DDPM

T = 11
B = 8
D = 16
x_0 = torch.randn((B,D)) * torch.randint(-5, 5, (B,D))
assert x_0.shape == (B, D)

eps_model = nn.Sequential(*[nn.Linear(16, 16)])
ddpm = DDPM(
    x_input_dim=16,
    log_every_t=1
)
def test_p_sample():

    pass
def test_q_sample():

    pass