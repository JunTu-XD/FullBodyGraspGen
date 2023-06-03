import functools

import torch
from matplotlib import pyplot as plt
from torch.optim import AdamW
from tqdm import tqdm

from WholeGraspPose.models.diffusion.Eps import Eps
from WholeGraspPose.models.diffusion.improved_diffusion.script_util import create_gaussian_diffusion

D = 16
B = 128
T = 1000

miu = torch.distributions.Normal(torch.zeros((D,))+0.1 , torch.ones((D,))).rsample()
sigma = 1
data_gen = lambda _=None: torch.distributions.Normal(miu, sigma).sample((B,))


gau_diff = create_gaussian_diffusion(steps=T)
model = Eps(D=D)

loss = gau_diff.training_losses(model, x_start=data_gen(), t=torch.randint(0, T, (B,)) )
print(loss)

opt = AdamW(model.parameters(), lr=0.001, weight_decay=0.8)

diff_loss = []
dists = []
for i in tqdm(range(3000)):
        opt.zero_grad()

        t=torch.randint(0, T, (B,))
        x_0 = data_gen()
        compute_losses = functools.partial(
                gau_diff.training_losses,
                model,
                x_0,
                t,

            )
        losses = compute_losses()
        loss = losses['loss'].mean()
        diff_loss.append(loss.detach())

        loss.backward()
        opt.step()


plt.title("diffusion loss")
plt.plot(diff_loss)
plt.show()

seq = gau_diff.p_sample_loop_progressive(model, shape=(3, D))
centers_seq = []

# eval_data = data_dict[names[3]][:, 0:16]
# eval_miu = data_dict[names[3]][:, 16:32]
# eval_var = data_dict[names[3]][:, 32:]
eval_miu = miu[None, :]

# min_indices = torch.argmin(torch.cdist(sample_x0, eval_miu), dim=1)
dists = []
# min_d =  torch.min(torch.cdist(sample_x0, data_dict[names[3]][:, 0:16]), dim=1)
# for _ in range(_num_draw):
#         centers_seq.append([])
for _s in seq:
        min_dist = (_s['sample'] - 1).norm(dim=1).mean()
        dists.append(min_dist.mean())
#     # centers_seq[_i].append(min_dist)
#     centers_seq[_i].append(torch.mean(_s, dim=0)[_i])


plt.plot(dists)
plt.hlines(y=(gau_diff.p_sample_loop(model, shape=(2, D)) - miu[None, :]).norm(dim=1).mean(), xmin=0, xmax=T, linestyles="--",
           label='l2 dist: data gen by miu and miu')
plt.grid()
plt.legend()
plt.show()