import argparse
import copy
import os
import sys

import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from absl import app, flags
from WholeGraspPose.models.diffusion.DDIM import DDIMSampler
from WholeGraspPose.models.diffusion.DDPM import DDPM
from WholeGraspPose.models.diffusion.DDPM_simple import GaussianDiffusionTrainer, GaussianDiffusionSampler
from WholeGraspPose.models.diffusion.Eps import Eps
from WholeGraspPose.models.models import FullBodyGraspNet
import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion

T = 500

B = 256
D = 64

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
    clip_denoised=True
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
FLAGS = flags.FLAGS
flags.DEFINE_bool('train', False, help='train from scratch')
flags.DEFINE_bool('eval', False, help='load ckpt.pt and evaluate FID and IS')
# UNet
flags.DEFINE_integer('ch', 128, help='base channel of UNet')
flags.DEFINE_multi_integer('ch_mult', [1, 2, 2, 2], help='channel multiplier')
flags.DEFINE_multi_integer('attn', [1], help='add attention to these levels')
flags.DEFINE_integer('num_res_blocks', 2, help='# resblock in each level')
flags.DEFINE_float('dropout', 0.1, help='dropout rate of resblock')
# Gaussian Diffusion
flags.DEFINE_float('beta_1', 1e-4, help='start beta value')
flags.DEFINE_float('beta_T', 0.02, help='end beta value')
flags.DEFINE_integer('T', 1000, help='total diffusion steps')
flags.DEFINE_enum('mean_type', 'epsilon', ['xprev', 'xstart', 'epsilon'], help='predict variable')
flags.DEFINE_enum('var_type', 'fixedlarge', ['fixedlarge', 'fixedsmall'], help='variance type')
# Training
flags.DEFINE_float('lr', 2e-4, help='target learning rate')
flags.DEFINE_float('grad_clip', 1., help="gradient norm clipping")
flags.DEFINE_integer('total_steps', 800000, help='total training steps')
flags.DEFINE_integer('img_size', 32, help='image size')
flags.DEFINE_integer('warmup', 5000, help='learning rate warmup')
flags.DEFINE_integer('batch_size', 128, help='batch size')
flags.DEFINE_integer('num_workers', 4, help='workers of Dataloader')
flags.DEFINE_float('ema_decay', 0.9999, help="ema decay rate")
flags.DEFINE_bool('parallel', False, help='multi gpu training')
# Logging & Sampling
flags.DEFINE_string('logdir', './logs/DDPM_CIFAR10_EPS', help='log directory')
flags.DEFINE_integer('sample_size', 64, "sampling size of images")
flags.DEFINE_integer('sample_step', 1000, help='frequency of sampling')
# Evaluation
flags.DEFINE_integer('save_step', 5000, help='frequency of saving checkpoints, 0 to disable during training')
flags.DEFINE_integer('eval_step', 0, help='frequency of evaluating model, 0 to disable during training')
flags.DEFINE_integer('num_images', 50000, help='the number of generated images for evaluation')
flags.DEFINE_bool('fid_use_torch', False, help='calculate IS and FID on gpu')
flags.DEFINE_string('fid_cache', './stats/cifar10.train.npz', help='FID cache')
FLAGS(sys.argv)

miu = torch.distributions.Normal(torch.zeros((D,)), torch.ones((D,))).rsample()
miu = miu / torch.max(torch.abs(miu))
sigma = torch.distributions.Normal(torch.sqrt(torch.ones((D,)) / 2), torch.ones((D,))).rsample()
sigma = sigma ** 2
data_gen = lambda _=None: torch.zeros((B,D))
recon_loss = torch.nn.MSELoss()


def ema(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(
            target_dict[key].data * decay +
            source_dict[key].data * (1 - decay))



def test_train():
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
    ema_model = ddpm
    for _i in tqdm(range(10000)):
        optimimzer.zero_grad()
        x_0 = data_gen()
        _loss, _ = ddpm.forward(x_0, condition=None)
        clip_grad_norm_(ddpm.model.parameters(), 1.0)
        _loss.backward()
        optimimzer.step()
        # ema(ddpm, ema_model, 0.7)
        if _i % 500 == 0:
            with torch.no_grad():
                bs = B
                shape = D
                ddpm_samples = ema_model.sample(batch_size=B, ddim=False)
                ddim_samples = ema_model.sample(batch_size=B, ddim=True)

                temp_loss = recon_loss(x_0, ddpm_samples)
                ddim_recon = recon_loss(x_0, ddim_samples)

                ddim_recon_seq.append(ddim_recon.detach())
                recon_seq.append(temp_loss.detach())
                diff_loss.append(_loss.detach())

        diffusion_lr_scheduler.step()

    print(recon_seq)

    print(ddim_recon_seq)
    print(diff_loss)

    _, seq = ema_model.sample(batch_size=128, ddim=False, return_intermediates=True)
    noise_seq = []
    for _s in reversed(seq):
        noise_seq.append(torch.mean(torch.norm(_s, dim=1)))
    plt.figure(0)
    plt.plot(list(range(len(recon_seq))), recon_seq)
    plt.show()

    plt.figure(1)
    plt.plot(list(range(len(noise_seq))), noise_seq)
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
    full_grasp_net.load_state_dict(torch.load(cfg.use_pretrained, map_location="cpu"), strict=False)
    full_grasp_net.adaptor.load_state_dict(torch.load(cfg.pretrained_adaptor, map_location="cpu"), strict=False)

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
    test_train()



