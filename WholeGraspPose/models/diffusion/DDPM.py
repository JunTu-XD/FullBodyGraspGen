"""
wild mixture of
https://github.com/lucidrains/denoising-diffusion-pytorch/blob/7706bdfc6f527f58d33f84b7b522e61e6e3164b3/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
https://github.com/openai/improved-diffusion/blob/e94489283bb876ac1477d5dd7709bbbd2d9902ce/improved_diffusion/gaussian_diffusion.py
https://github.com/CompVis/taming-transformers
-- merci
"""

import torch
import torch.nn as nn
import numpy as np
from torch import clamp
from torch.optim.lr_scheduler import LambdaLR
from einops import rearrange, repeat
from contextlib import contextmanager
from functools import partial
from tqdm import tqdm
from torchvision.utils import make_grid

from WholeGraspPose.models.diffusion.Eps import Eps
from WholeGraspPose.models.diffusion.utils import exists, default, mean_flat, count_params, instantiate_from_config
from WholeGraspPose.models.diffusion.ema import LitEma
from WholeGraspPose.models.diffusion.distributions import normal_kl, DiagonalGaussianDistribution
from WholeGraspPose.models.diffusion.modules_util import make_beta_schedule, extract_into_tensor, noise_like
from WholeGraspPose.models.diffusion.DDIM import DDIMSampler

__conditioning_keys__ = {'concat': 'c_concat',
                         'crossattn': 'c_crossattn',
                         'adm': 'y'}


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


def uniform_on_device(r1, r2, shape, device):
    return (r1 - r2) * torch.rand(*shape, device=device) + r2


class DDPM(nn.Module):
    # classic DDPM with Gaussian diffusion
    def __init__(self,
                 model=None,
                 timesteps=500,
                 beta_schedule="linear",
                 loss_type="l2",
                 ckpt_path=None,
                 ignore_keys=[],
                 load_only_unet=False,
                 monitor="val/loss",
                 use_ema=False,
                 x_dim=16,
                 log_every_t=100,
                 clip_denoised=True,
                 linear_start=1e-4,
                 linear_end=2e-2,
                 cosine_s=8e-3,
                 given_betas=None,
                 original_elbo_weight=0.,
                 v_posterior=0.,  # weight for choosing posterior variance as sigma = (1-v) * beta_tilde + v * beta
                 l_simple_weight=1.,
                 parameterization="eps",  # all assuming fixed variance schedules
                 scheduler_config=None,
                 use_positional_encodings=False,
                 learn_logvar=False,
                 logvar_init=0.,
                 w=4,
                 classifier_free_cond_dropout=0.2
                 ):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        assert parameterization in ["eps", "x0"], 'currently only supporting "eps" and "x0"'
        self.parameterization = parameterization
        print(f"{self.__class__.__name__}: Running in {self.parameterization}-prediction mode")
        self.cond_stage_model = None
        self.clip_denoised = clip_denoised
        self.log_every_t = log_every_t
        self.x_dim = x_dim
        self.use_positional_encodings = use_positional_encodings
        self.model = Eps(D=self.x_dim)
        self.w = w
        self.classifier_free_cond_dropout = classifier_free_cond_dropout
        count_params(self.model, verbose=True)
        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self.model)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        self.use_scheduler = scheduler_config is not None
        if self.use_scheduler:
            self.scheduler_config = scheduler_config

        self.v_posterior = v_posterior
        self.original_elbo_weight = original_elbo_weight
        self.l_simple_weight = l_simple_weight

        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys, only_model=load_only_unet)

        self.register_schedule(given_betas=given_betas, beta_schedule=beta_schedule, timesteps=timesteps,
                               linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s)

        self.loss_type = loss_type

        self.learn_logvar = learn_logvar
        self.logvar = torch.full(fill_value=logvar_init, size=(self.num_timesteps,), device=self.device)
        if self.learn_logvar:
            self.logvar = nn.Parameter(self.logvar, requires_grad=True)

    def register_schedule(self, given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        """
        for alpha_bar at each time t. Ref. LDM codebase
       """
        if exists(given_betas):
            betas = given_betas
        else:
            betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end,
                                       cosine_s=cosine_s)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))  # alpha_bar
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (
                1. - alphas_cumprod) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

        if self.parameterization == "eps":
            lvlb_weights = self.betas ** 2 / (
                    2 * self.posterior_variance * to_torch(alphas) * (1 - self.alphas_cumprod))
        elif self.parameterization == "x0":
            lvlb_weights = 0.5 * np.sqrt(torch.Tensor(alphas_cumprod)) / (2. * 1 - torch.Tensor(alphas_cumprod))
        else:
            raise NotImplementedError("mu not supported")
        # TODO how to choose this term
        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer('lvlb_weights', lvlb_weights, persistent=False)
        assert not torch.isnan(self.lvlb_weights).all()

    @contextmanager
    def ema_scope(self, context=None):
        """
        Exponential moving average.
        model_ema. replace weights in model with EMA.
        info https://www.reddit.com/r/StableDiffusion/comments/x5am4v/ema_model_vs_non_ema_differences/
        """
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def q_mean_variance(self, x_start, t):
        """
        mean, var of q(x_t | x_0)
        :param
        x_0: [B, latent D], noiseless
        t: time step number.
        :return:
        (mean, variance, log_variance)
        """
        mean = (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start)
        variance = extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract_into_tensor(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        """
        rewrite p(x_t | x_0) to x_t => x_0
        x_0 = 1/sqrt(alpha_bar) * x_t - noise * 1/sqrt(1-alpha_bar)
        :param x_t:
        :param t:
        :param noise:
        :return: x_0
        """

        return (
                extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        """
        posterior(x_t-1 | x_t, x_0)
        :return:
        posterior_mean, posterior_variance, posterior_log_variance_clipped
        """
        posterior_mean = (
                extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, condition, clip_denoised: bool):
        """
        p(x_t-1 | x_t, x_0, t)
        where x_0 = reconstruct by (self.model out .e.g. noise, x_t. t)
        :param condition:
        :param x:
        :param t:
        :param clip_denoised:
        :return:
        model_mean, posterior_variance, posterior_log_variance
        """
        pred_eps_cond = self.model(x=x, t=t, condition=condition)
        _condition = torch.zeros_like(condition, device=self.device)
        pred_eps_uncond = self.model(x=x, t=t, condition=_condition)
        pred_eps = (1 + self.w) * pred_eps_cond - self.w * pred_eps_uncond

        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=pred_eps)
        elif self.parameterization == "x0":
            raise NotImplementedError()

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, condition=None, clip_denoised=True, repeat_noise=False):
        b, *_, device = *x.shape, x.device
        model_mean, model_variance, model_log_variance = self.p_mean_variance(x=x, t=t, condition=condition,
                                                                              clip_denoised=clip_denoised)
        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, condition, return_intermediates=False):
        device = self.betas.device
        b = shape[0]
        x_t_minus_1 = torch.randn(shape, device=device)
        intermediates = [x_t_minus_1]
    
        for i in reversed(range(0, self.num_timesteps)):
            x_t_minus_1 = self.p_sample(x_t_minus_1, torch.full((b,), i, device=device, dtype=torch.long),
                                        condition=condition,
                                        clip_denoised=self.clip_denoised)
            if i % self.log_every_t == 0 or i == self.num_timesteps - 1:
                intermediates.append(x_t_minus_1)
        ## now it's x_0
        if return_intermediates:
            return x_t_minus_1, intermediates
        return x_t_minus_1

    @torch.no_grad()
    def sample(self, batch_size=16, condition=None, return_intermediates=False, ddim=True, ddim_steps=200, **kwargs):
        x_dim = self.x_dim
        if ddim:
            ddim_sampler = DDIMSampler(self)
            shape = x_dim
            samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size,
                                                         shape, condition, verbose=False, **kwargs)
            if return_intermediates:
                return samples, intermediates
            return samples
        else:
            return self.p_sample_loop((batch_size, x_dim), condition,
                                      return_intermediates=return_intermediates)

    def q_sample(self, x_start, t, noise=None):
        """
        draw sample from p(x_t|x_0)
        :param x_start:
        :param t:
        :return: mean + noise~N(0,1) * var
        sqrt(alpha_bar) * x_0 + noise * sqrt(1/alpha_bar -1 )
        """
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

    def get_loss(self, pred, target, mean=True):
        if self.loss_type == 'l1':
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif self.loss_type == 'l2':
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred)
            else:
                loss = torch.nn.functional.mse_loss(target, pred, reduction='none')
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")

        return loss

    def p_losses(self, x_start, t, condition=None, noise=None, **kwargs):
        """
        LDM Appendix B. Detailed Information on Denoising Diffusion Mod
        :param condition:
        :param x_start:
        :param t:
        :param noise: default N(0,1). noise at time t.
        :return: loss, loss_dict
        """
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        B = x_noisy.shape[0]
        condition[torch.rand(B) < self.classifier_free_cond_dropout] = 0

        model_output = self.model(x_noisy, t=t, condition=condition)

        loss_dict = {}
        prefix = 'diffusion_train' if self.training else 'diffusion_val'

        if self.parameterization == "x0":
            raise NotImplementedError()
        elif self.parameterization == "eps":
            target = noise
        else:
            raise NotImplementedError()

        loss_simple = self.get_loss(model_output, target, mean=False).mean([1])
        loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})

        logvar_t = self.logvar[t].to(self.device)
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        # loss = loss_simple / torch.exp(self.logvar) + self.logvar
        if self.learn_logvar:
            loss_dict.update({f'{prefix}/loss_gamma': loss.mean()})
            loss_dict.update({'logvar': self.logvar.data.mean()})

        loss = self.l_simple_weight * loss.mean()

        loss_vlb = self.get_loss(model_output, target, mean=False).mean(dim=(1))
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        loss_dict.update({f'{prefix}/loss_vlb': loss_vlb})
        loss += (self.original_elbo_weight * loss_vlb)
        loss_dict.update({f'diffusion_loss_total': loss})

        return loss, loss_dict

    def construct_condition(self, obj_feature, transf_transl=None, obj_xyz=None, **kwargs):
        condition = obj_feature
        return condition

    def forward(self, x, condition, *args, **kwargs):
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        return self.p_losses(x, t, condition=condition, *args, **kwargs)



