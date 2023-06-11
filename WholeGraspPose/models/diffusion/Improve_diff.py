import torch.distributions
from tqdm import tqdm

from models.diffusion.Eps import Eps
from models.diffusion.improved_diffusion.script_util import create_gaussian_diffusion
from models.diffusion.improved_diffusion.train_util import TrainLoop

D = 16
B = 64
mu = torch.ones((D, )) * 0.2
var = torch.ones((D, ))
data_gen = lambda _=None: torch.distributions.Normal(mu, var).rsample((B, ))
def test_diff():
    gau_diff = create_gaussian_diffusion(
        steps=1000,
        learn_sigma=False,
        sigma_small=True,
        noise_schedule="linear",
        use_kl=True,
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
        timestep_respacing="",
    )
    model = Eps(D=D)
    tl = TrainLoop(
        model=model,
        diffusion=gau_diff,
        # data=[],
        batch_size=B,
        microbatch=-1,
        lr=1e-4,
        ema_rate="0.9999",
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        schedule_sampler="uniform",
        weight_decay=0.0,
        lr_anneal_steps=0,
    )
    for _i in tqdm(range(100000)):
        batch_data = data_gen()
        tl.run_step(batch=batch_data, cond=None)
    sample = gau_diff.ddim_sample_loop(model, shape = (D,))
    print(sample)

if __name__ =="__main__":
    test_diff()