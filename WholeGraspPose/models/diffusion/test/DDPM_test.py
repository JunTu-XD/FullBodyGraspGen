import torch
from torch import nn

from WholeGraspPose.models.diffusion.DDPM import DDPM

T = 11
B = 32
D = 16
x_0 = torch.randn((B, D)) * torch.randint(-5, 5, (B, D))
assert x_0.shape == (B, D)


class T_EPS(nn.Module):
    def __init__(self, D):
        super(T_EPS, self).__init__()
        self.model = nn.Sequential(*[nn.Linear(D, D)])

    def forward(self, x, t):
        return self.model(x)


ddpm = DDPM(
    timesteps = T,
    model=T_EPS(D=D),
    x_dim=16,
    log_every_t=1
)


def test_p_sample():
    x_0, reversed_out = ddpm.sample(batch_size=B, return_intermediates=True)
    assert len(reversed_out) == (T+1)


def test_q_sample():
    x_t = ddpm.q_sample(x_0, t=torch.randint(0,T, (B,)))
    assert x_t.shape == x_0.shape

def test_train():
    loss = ddpm.training_step(x_0)
    assert loss

