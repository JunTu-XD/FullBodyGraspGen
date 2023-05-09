import torch
from torch import nn


class DDPM(nn.Module):
    """
    mainly follow DDPM in LDM manner.
    """

    def register_schedule(self, given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        """
        for alpha_bar at each time t.
        :param given_betas:
        :param beta_schedule:
        :param timesteps:
        :param linear_start:
        :param linear_end:
        :param cosine_s:
        :return:

        """
        pass

    def q_mean_var(self, x_0, t):
        """
        mean, var of q(x_t | x_0)
        :param
        x_0: [B, latent D], noiseless
        t: time step number.
        :return:
        (mean, variance, log_variance)
        """
        pass

    @torch.no_grad()
    def p_mean_variance(self, x, t, clip_denoised: bool):
        """

        :param x:
        :param t:
        :param clip_denoised:
        :return:
        """
        pass

    @torch.no_grad()
    def p_sample(self, x_t):
        """
        Denoising. Reversed process
        :param

        :return
        x_(t-1)
        """
        pass

    def predict_start_from_noise(self, x_t, t, noise):
        """

        :param x_t:
        :param t:
        :param noise:
        :return:
        """
    def q_posterior(self, x_start, x_t, t):
        """
        posterior(x_t-1 | x_t,
        :return:
        posterior_mean, posterior_variance, posterior_log_variance_clipped
        """


    def p_mean_variance(self, x_t, t, clip_denoised: bool):
        """
        p(x_t-1 | x_t, x_0, t)
        where x_0 = reconstruct by (self.model out .e.g. noise, x_t. t)
        :param x:
        :param t:
        :param clip_denoised:
        :return:
        model_mean, posterior_variance, posterior_log_variance
        """
        model_out = self.model(x_t, t)
        # model out -> reconstruct x_0
        if self.parameterization == "eps":
            ## eps predict noise
            x_recon = self.predict_start_from_noise(x_t, t=t, noise=model_out)
        elif self.parameterization == "x0":
            ## predict x_0
            x_recon = model_out
        else:
            raise NotImplementedError(f"reconstruction y parameterization-[{self.parameterization}] not implemented")

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    def loss(self, pred, target, mean=True):
        """
        loss func wrapper. L1, L2.
        :param mean: return mean of losses
        :return:
        loss: Tensor of loss.
        """
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

    def p_losses(self, x_0, t, noise=None):
        """
        :param x_0:
        :param t:
        :param noise:
        :return: loss, loss_dict
        """
        loss_dict = {}
        if self.parameterization == "eps":
            target = noise
        elif self.parameterization == "x0":
            target = x_0
        else:
            raise NotImplementedError(f"Paramterization {self.parameterization} not ye

    def forward(self):

        pass
