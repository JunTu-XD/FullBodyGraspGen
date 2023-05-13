import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from WholeGraspPose.models.diffusion.DDPM import DDPM
from WholeGraspPose.models.diffusion.DDIM import DDIMSampler

T = 11
B = 32
D = 512
S = 4 # DDIM sampling steps
cond_dim = 518
x_0 = torch.randn((B, D)) * torch.randint(-5, 5, (B, D))
cond = torch.randn((B, cond_dim)) # embedding of the condition


ddpm = DDPM(
    timesteps = T,
    x_dim=D,
    log_every_t=1
)
ddpm.learning_rate = 0.001

# def test_p_sample():
#     x_0, reversed_out = ddpm.sample(batch_size=B, return_intermediates=True)
#     assert len(reversed_out) == (T+1)
#
#
# def test_q_sample():
#     x_t = ddpm.q_sample(x_0, t=torch.randint(0,T, (B,)))
#     assert x_t.shape == x_0.shape
#
# def test_train_step():
#     loss = ddpm.training_step(x_0, batch_idx=-1)
#     assert loss
#
# def test_val():
#     ddpm.validation_step(x_0, batch_idx=-1)

def test_ddim_sampling():
    ddim = DDIMSampler(ddpm)
    bs = x_0.shape[0]
    shape = x_0.shape[1]
    samples, intermediates = ddim.sample(S=S, conditioning = cond, batch_size=bs, shape=shape, eta=1., verbose=False)
    print(samples.shape) # (B, D)


