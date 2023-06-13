import torch
from torch.utils.data import TensorDataset
from tqdm import tqdm

from models.diffusion.DDPM import DDPM
from models.diffusion.Eps import Eps
from utils.cfg_parser import Config

from torch.utils.tensorboard import SummaryWriter
class DiffusionTrainer:
    def __init__(self, cfg):
        self.device = torch.device("cuda:%d" % cfg.cuda_id if torch.cuda.is_available() else "cpu")
        self.cfg = cfg
        self.writer = SummaryWriter(f"{cfg.save_folder}/tb/")
        self.train_tensor = self.load_data()
        self.train_loader = torch.utils.data.DataLoader(self.train_tensor,
                                                    shuffle=True,
                                                    drop_last=True,
                                                    batch_size=cfg.batch_size)

        self.model = Eps(D=cfg.x_dim)
        if self.cfg.trained_diffusion is not None:
            self.load_model()
        self.diffusion = DDPM()
        self.diffusion.model = self.model

        self.optimizer = torch.optim.AdamW(self.diffusion.model.parameters(), lr=cfg.lr)


    def load_data(self):
        data_dict = torch.load(self.cfg.dataset_path, map_location=self.device)
        return TensorDataset(*[data_dict['mu'], data_dict['var'], data_dict['label']])

    def load_model(self):
        if self.cfg.trained_model is not None:
            self.model.load_state_dict(cfg.trained_model)

    def train_ddpm(self):
        for e in range(self.cfg.epoch):
            for _i, (_mu, _var, _label) in enumerate(tqdm(self.train_loader)):
                self.optimizer.zero_grad()

                x_0 = torch.distributions.Normal(_mu, _var).rsample()
                loss, loss_dict = self.diffusion(x=x_0, condition=_label)
                loss.backward()

                self.writer.add_scalar("loss/diffusion_l2", loss, _i + e *len(self.train_loader) )
                self.writer.flush()

                self.optimizer.step()

    def load_model(self):
        self.model.load_state_dict(torch.load(self.cfg.trained_diffusion, map_location=self.device), strict=False)

    def dist_metrics(self, p=2):
        sample_batch = 512

        samples = []
        mean_samples = []
        for label in range(2):
            cond_ = torch.nn.functional.one_hot(torch.ones((sample_batch, )).long() * label, 23).float()
            samples_ = self.diffusion.sample(batch_size=sample_batch, ddim=False, condition= cond_)
            samples.append(samples_)
            mean_samples.append(samples_.mean(dim=0)[None, :])

        internal_dist = []
        for samples_l in samples:
            dist = torch.cdist(samples_l, samples_l, p = p)
            mean_internal_dist = torch.sum(dist) / (sample_batch**2 - samples)
            internal_dist.append(mean_internal_dist)
        mean_samples = torch.cat(mean_samples)
        external_dist = torch.cdist(mean_samples)
        return

    def save(self):
        torch.save(self.model.state_dict(), f"{self.cfg.save_folder}/diffusion_model.pt")

    def original_data(self):
        data_dict = torch.load("dataset/saga_male_latent_label.pt", map_location=self.device)
        label_inds = torch.argmax(data_dict['label'], dim=1)
        labels_mu = {i:[] for i in range(22)}
        labels_var = {i:[] for i in range(22)}

        mius = [[] for i in range(22)]
        for i in range(data_dict['mu'].shape[0]):
            mu = data_dict['mu'][i]
            var = data_dict['var'][i]
            labels_mu[int(label_inds[i])].append(mu)
            labels_var[int(label_inds[i])].append(var)
        mius_tensor = []
        for i, m in labels_mu.items():
            mius.append(torch.cat(m).mean(dim=0))
        return
if __name__=="__main__":
    exp_name = "test"

    cfg = Config(**{
        "exp_name": exp_name,
        "dataset_path":'dataset/saga_male_latent_label.pt',
        "batch_size": 64,
        "x_dim":16,
        "trained_model":None,
        "epoch":2,
        "save_folder":f"logs/{exp_name}/",
        "lr":1e-4,
        "trained_diffusion": f"logs/{exp_name}/diffusion_model.pt",
    })

    trainer = DiffusionTrainer(cfg)
    # trainer.train_ddpm()
    trainer.dist_metrics()
    # trainer.save()
    trainer.original_data()
