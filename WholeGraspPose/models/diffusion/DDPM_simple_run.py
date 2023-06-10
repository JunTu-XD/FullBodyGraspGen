import argparse
import copy
import os
import sys
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('macosx')
import torch.utils.data as td
import torch
from torch import nn, optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from absl import app, flags
from WholeGraspPose.models.diffusion.DDIM import DDIMSampler
from WholeGraspPose.models.diffusion.DDPM import DDPM
from WholeGraspPose.models.diffusion.Eps import Eps

from WholeGraspPose.models.models import FullBodyGraspNet
import torch

writer = SummaryWriter(log_dir="./test/logs")
# writer.add_scalar("loss", loss )
# writer.close()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

T = 1000

B = 128
D = 16

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
    timesteps=T,
    model=Eps(D=D),
    x_dim=D,
    log_every_t=1,
    parameterization='eps',
    # linear_start = 1e-3,
    # linear_end = 5e-2
    # clip_denoised=True,
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

pos_miu = torch.ones((D,))  # torch.distributions.Normal(torch.zeros((D,))+0.4 , torch.ones((D,))*0.2 ).rsample()
neg_miu = torch.ones((D,))  # torch.distributions.Normal(torch.zeros((D,))-0.4 , torch.ones((D,))*0.2 ).rsample()
sigma = 0.2
data_gen = lambda miu: torch.distributions.Normal(miu, sigma).sample((int(B/2), ))

recon_loss = torch.nn.MSELoss()



def ema(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(
            target_dict[key].data * decay +
            source_dict[key].data * (1 - decay))



def plot_ddpm_alpha_beta():

    plt.plot(ddpm.sqrt_alphas_cumprod,label="sqrt_alphas_cumprod")
    plt.plot(ddpm.sqrt_one_minus_alphas_cumprod, label="sqrt_one_minus_alphas_cumprod")
    plt.legend()
    plt.show()

    plt.plot(ddpm.log_one_minus_alphas_cumprod, label = "log_one_minus_alphas_cumprod")
    plt.plot(ddpm.sqrt_recipm1_alphas_cumprod, label = "sqrt_recipm1_alphas_cumprod")
    plt.plot(ddpm.sqrt_recip_alphas_cumprod, label="sqrt_recip_alphas_cumprod")
    plt.legend()
    plt.show()

def test_train():
    # data_dict = torch.load("saga_male_latent_label.pt", map_location=device)
    # names = list(data_dict.keys())
    # train_set = td.TensorDataset(data_dict[names[0]], data_dict[names[1]])
    # eval_set = td.TensorDataset(data_dict[names[2]], data_dict[names[3]])

    batch_s = 128
    # saga_train_loader = td.DataLoader(train_set, shuffle=True, drop_last=True, batch_size=batch_s)
    # saga_eval_loader = td.DataLoader(eval_set, shuffle=False, drop_last=False, batch_size=batch_s)

    ddim = DDIMSampler(ddpm)

    optimimzer = torch.optim.AdamW(ddpm.parameters(), lr=0.01)
    diffusion_lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimimzer,
                                                                            T_0=5000,
                                                                            # Number of iterations for the first restart
                                                                            T_mult=1,
                                                                            # A factor increases TiTi​ after a restart
                                                                            eta_min=1e-5)
    ddpm.train()
    recon_seq = []
    ddim_recon_seq = []
    diff_loss = []
    for _i in tqdm(range(10000)):
            optimimzer.zero_grad()

            x_1 = data_gen(pos_miu)
            x_0 = data_gen(neg_miu)

            label_0 = torch.zeros((int(B/2), 2))
            label_0[:, 0] = 1
            label_1 = torch.zeros((int(B/2), 2))
            label_1[:, 1] = 1
            x = torch.cat([x_0,x_1], dim=0)
            label = torch.cat([label_0, label_1], dim=0)
            perm = torch.randperm(x.shape[0])
            x = x[perm, :]
            label = label[perm, :]
            # x_0 = t_label
            # x_0 = torch.distributions.Normal(t_mu, t_var).sample()

            _loss, _ = ddpm(x, condition=label)
            clip_grad_norm_(ddpm.model.parameters(), 1.0)
            _loss.backward()
            optimimzer.step()
            diff_loss.append(_loss.detach())

            writer.add_scalar("loss", _loss.detach(), _i)
            # ema(ddpm, ema_model, 0.7)
            if _i % 500 == 0:
                with torch.no_grad():
                    bs = B
                    shape = D
                    # ddpm_samples = ddpm.sample(batch_size=B, ddim=False)
                    # ddim_samples = ddpm.sample(batch_size=B, ddim=True)
                    #

                    # ddim_recon = recon_loss(x_0, ddim_samples)
                    #
                    # ddim_recon_seq.append(ddim_recon.detach())
                    # recon_seq.append(temp_loss.detach())
                    # diff_loss.append(_loss.detach())

            diffusion_lr_scheduler.step()

    _num_draw = 4
    sample_num = 512
    cond = torch.zeros((sample_num, 2))
    cond[:, 0] += 1
    sample_x0, seq = ddpm.sample(batch_size=sample_num, ddim=False, return_intermediates=True,
                                 condition=cond)
    x0_cond_0_min_dist= torch.mean(torch.cdist(sample_x0, neg_miu[None, :]))
    x0_cond_1_min_dist = torch.mean(torch.cdist(sample_x0, pos_miu[None, :]))

    cond = torch.zeros((sample_num, 2))
    cond[:, 1] += 1
    # sample_x1, seq = ddpm.sample(batch_size=sample_num, ddim=False, return_intermediates=True,
    #                              condition=cond)
    #
    # x1_cond_0_min_dist = torch.mean(torch.cdist(sample_x1, neg_miu[None, :]))
    # x1_cond_1_min_dist = torch.mean(torch.cdist(sample_x1, pos_miu[None, :]))
    writer.close()
    dists = []

    # for _i, _s in enumerate(reversed(seq)):
            # min_dist_0 = (_s - ).norm(dim=1)
            # min_dist_1 = (_s - ).norm(dim=1)
            # writer.add_scalar("dist_diff_0and1",  torch.abs(min_dist_1.mean()- min_dist_0.mean()), _i)
            # writer.add_scalar("min_dist", torch.min(min_dist_1.mean(), min_dist_0.mean()), _i)




import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

def one_d_diff():
    x0 = -5  # "realistic" sample which the model needs to learn to generate
    n_steps = 100

    alphas = 1. - torch.linspace(0.001, 0.2, n_steps)
    alphas_cumprod = torch.cumprod(alphas, axis=0)

    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - sqrt_alphas_cumprod ** 2)

    def q_sample(x_0, t, noise):
        """
        Sample x at time t given the value of x at t=0 and the noise
        """
        return sqrt_alphas_cumprod.gather(-1, t) * x_0 + sqrt_one_minus_alphas_cumprod.gather(-1, t) * noise

    def recon_x_0(x_t, t, noise):
        return 1/sqrt_alphas_cumprod.gather(-1, t) * (x_t - (sqrt_one_minus_alphas_cumprod.gather(-1, t) ) * noise)

    for t in [1, n_steps // 10, n_steps // 2, n_steps - 1]:
        noised_x = q_sample(x0, torch.tensor(t), torch.randn(1000))
        plt.hist(noised_x.numpy(), bins=100, alpha=0.5, label=f"t={t}");
    plt.legend()
    plt.show()

    res = [(t, q_sample(x0, torch.tensor(t), torch.randn(1)).item()) for _ in range(10) for t in range(n_steps)]
    x, y = list(zip(*res))
    plt.scatter(x, y, s=1)
    plt.xlabel("time")
    plt.ylabel("x")
    plt.show()

    class DenoiseModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a = torch.nn.Parameter(torch.tensor(1.))
            self.b = torch.nn.Parameter(torch.tensor(0.))
            self.c = torch.nn.Parameter(torch.tensor(0.))

        def forward(self, x, t):
            return self.a * x + self.b * t + self.c

    denoise = DenoiseModel()
    optimizer = torch.optim.Adam(denoise.parameters())

    def p_loss(x, t):
        # Generate a noise
        noise = torch.randn(t.shape)
        # Compute x at time t with this value of the noise - forward process
        noisy_x = q_sample(x, t, noise)
        # Use our trained model to predict the value of the noise, given x(t) and t
        noise_computed = denoise(noisy_x, t)
        # Compare predicted value of the noise with the actual value
        return F.mse_loss(noise, noise_computed)

    n_epochs = 10000
    batch_size = 1000
    for step in range(n_epochs):
        optimizer.zero_grad()
        t = torch.randint(0, n_steps, (batch_size,))  # Pick random time step
        loss = p_loss(x0, t)
        loss.backward()
        if step % (n_epochs // 10) == 0:
            print(
                f"loss={loss.item():.4f}; a={denoise.a.item():.4f}, b={denoise.b.item():.4f}, c={denoise.c.item():.4f}")
        optimizer.step()
    print(
        f"final: loss={loss.item():.4f}; a={denoise.a.item():.4f}, b={denoise.b.item():.4f}, c={denoise.c.item():.4f}")

    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    posterior_variance = (1 - alphas) * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

    plt.plot(sqrt_alphas_cumprod, label="sqrt_alphas_cumprod")
    plt.plot(sqrt_one_minus_alphas_cumprod, label="sqrt_one_minus_alphas_cumprod")
    plt.legend()
    plt.show()

    def p_sample(x, t):
        """
        One step of revese process sampling - Algorithm 2 from the paper
        """
        alpha_t = alphas.gather(-1, t)
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod.gather(-1, t)
        alphas_cumprod_t = alphas_cumprod.gather(-1, t)
        # Get mean x[t - 1] conditioned at x[t] - see eq. (11) in the paper
        noise = torch.randn((1,))
        model_mean = torch.sqrt(1.0 / alpha_t) * (x - (1 - alpha_t) * noise / sqrt_one_minus_alphas_cumprod_t)

        model_mean2 = 1/(1-alphas_cumprod_t) * torch.sqrt(alphas_cumprod_prev[t]) * (1-alpha_t) * recon_x_0(x, t, noise=noise) \
                      + torch.sqrt(alpha_t) * (1-alphas_cumprod_prev[t]) / (1-alphas_cumprod_t) * x
        # Get variance of x[t - 1]
        model_var = posterior_variance.gather(-1, t)
        # Samples for the normal distribution with given mean and variance
        return model_mean2 + torch.sqrt(model_var) * torch.randn(1)

    plt.figure(figsize=(10, 10))
    for _ in range(5):
        x_gens = []
        x_means = []
        x_gen = torch.randn((1,))
        for i in range(n_steps - 1, 0, -1):
            x_gen = p_sample(x_gen, torch.tensor(i))
            x_gens.append(x_gen.detach().numpy()[0])
            # x_means.append(x_mean.detach().numpy()[0])
        plt.plot(x_gens[::-1])
        plt.plot(x_means[::-1])
    plt.hlines(x0, 0, 100, color="black", linestyle="--", label=f"x0 = {x0}")
    plt.hlines(0, 0, 100, color="r", linestyle="--")
    plt.legend(loc="upper left")
    plt.title("Reverse process - denoising")
    plt.show()



    x_gen = torch.randint(-100, 100, (10000,))
    for i in range(n_steps - 1, 0, -1):
        # denoise the sample step by step backawards in time
        x_gen = p_sample(x_gen, torch.tensor(i))
        if i % 10 == 0:
            plt.hist(x_gen.detach().numpy(), range=(-100, 100), bins=100)
            plt.show()


def test_FBGrasp_Load():
    cwd = os.getcwd()
    default_cfg_path = 'WholeGraspPose/configs/WholeGraspPose.yaml'

    cfg = {
        'batch_size': 64,
        'n_workers': 2,
        'use_multigpu': False,
        'kl_coef': 0.5,
        'dataset_dir': "/cluster/work/cvl/wuyan/data/GRAB-series/GrabPose_r_fullbody/data",
        'base_dir': cwd,
        'work_dir': "./",
        'base_lr': 5e-4,
        'best_net': None,
        'gender': None,
        'exp_name': "test_exp",
    }

    cfg = Config(default_cfg_path=default_cfg_path, **cfg)
    full_grasp_net = FullBodyGraspNet(cfg)
    check_params = torch.detach_copy(full_grasp_net.marker_net.enc_rb1.fc1.weight)
    check_ada_params = torch.detach_copy(full_grasp_net.adaptor.fc1.weight)
    full_grasp_net.load_state_dict(torch.load(cfg.use_pretrained, map_location=device), strict=False)
    full_grasp_net.adaptor.load_state_dict(torch.load(cfg.pretrained_adaptor, map_location=device), strict=False)

    load_params = torch.detach_copy(full_grasp_net.marker_net.enc_rb1.fc1.weight)
    load_ada_params = torch.detach_copy(full_grasp_net.adaptor.fc1.weight)
    assert (check_params - load_params).norm() > 0
    assert (check_ada_params - load_ada_params).norm() > 0

    vars_net = [var[1] for var in full_grasp_net.named_parameters()]

    net_n_params = sum(p.numel() for p in vars_net if p.requires_grad)
    assert net_n_params == (sum(p.numel() for p in full_grasp_net.diffusion_parameters()) + sum(
        p.numel() for p in full_grasp_net.encoder_decoder_parameters()))
    print('\nTotal Trainable Parameters for ContactNet is %2.2f M.' % ((net_n_params) * 1e-6))


if __name__ == "__main__":
    # one_d_diff()
    # plot_ddpm_alpha_beta()
    # tb = get_timestep_embedding(torch.tensor([1,100,1000]), 16)
    # test_train()

    # plt.plot([1,2,3])
    # plt.ylabel("L2 distance between sampled x_t mean \nand original distribution mean.")
    # plt.show()
    cls = {0:[], 1:[]}
    data_dict = torch.load("saga_male_latent_label.pt", map_location=device)
    for i in range(data_dict['mu'].shape[0]):
        mu = data_dict['mu'][i]
        var = data_dict['var'][i]
        label = data_dict['label'][i]