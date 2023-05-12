import torch
from pytorch_lightning import Trainer
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from WholeGraspPose.models.diffusion.DDPM import DDPM
from WholeGraspPose.models.diffusion.Eps import Eps

T = 11

B = 32
D = 512

x_0 = torch.randn((B, D)) * torch.randint(-3, 3, (B, D))
t = torch.randint(0, T, (B,))

assert x_0.shape == (B, D)

train_data = []
for _ in range(100):
    train_data.append(x_0)
train_data = torch.cat(train_data, dim=0)
train_data = TensorDataset(train_data)
train_loader = DataLoader(dataset=train_data, batch_size=B)

ddpm = DDPM(
    timesteps = T,
    model=Eps(D=D),
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

def test_train():
    loss, _ = ddpm.p_losses(x_0, t, condition=torch.randn((B, D)))
    loss_1, _ = ddpm.forward(x_0, condition=torch.randn((B, D)))

    optim = torch.optim.AdamW(ddpm.parameters(), lr=0.01)
    ddpm.train()

    for _ in tqdm(range(100)):
        optim.zero_grad()
        _loss, _ = ddpm.forward(x_0, condition=torch.randn((B, D)))
        _loss.backward()
        optim.step()

    loss_2, _ = ddpm.forward(x_0, condition=torch.randn((B, D)))
    print (f"\nloss delta {(loss - loss_2)}")
    assert (loss - loss_2) > torch.abs(loss-loss_1)

