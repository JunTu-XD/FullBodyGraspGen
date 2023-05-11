import torch
from pytorch_lightning import Trainer
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from WholeGraspPose.models.diffusion.DDPM import DDPM

T = 11
B = 32
D = 16
x_0 = torch.randn((B, D)) * torch.randint(-5, 5, (B, D))
assert x_0.shape == (B, D)

train_data = []
for _ in range(100):
    train_data.append(x_0)
train_data = torch.cat(train_data, dim=0)
train_data = TensorDataset(train_data)
train_loader = DataLoader(dataset=train_data, batch_size=B)

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

def test_train():
    loss = ddpm.training_step(x_0, batch_idx=-1)
    t = Trainer(max_epochs=20)
    t.fit(ddpm, train_loader)
    # loss_2 = ddpm.training_step(x_0, batch_idx=-1)
    print(f"loss {loss} loss2 {ddpm(x_0)}")

