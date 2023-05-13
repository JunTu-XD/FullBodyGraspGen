import argparse
import os

import torch
from pytorch_lightning import Trainer
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from WholeGraspPose.models.diffusion.DDPM import DDPM
from WholeGraspPose.models.diffusion.Eps import Eps
from WholeGraspPose.models.models import FullBodyGraspNet
from utils.cfg_parser import Config

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

def test_FBGrasp_Load():
    cwd = os.getcwd()
    default_cfg_path = 'WholeGraspPose/configs/WholeGraspPose.yaml'

    cfg = {
        'batch_size':64,
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

    vars_net = [var[1] for var in full_grasp_net.named_parameters()]

    net_n_params = sum(p.numel() for p in vars_net if p.requires_grad)
    assert net_n_params == (sum(p.numel() for p in full_grasp_net.diffusion_parameters()) + sum(
        p.numel() for p in full_grasp_net.encoder_decoder_parameters()))
    print('\nTotal Trainable Parameters for ContactNet is %2.2f M.' % ((net_n_params) * 1e-6))