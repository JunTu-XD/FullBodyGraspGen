import torch
from torch.utils.data import TensorDataset
from tqdm import tqdm

from WholeGraspPose.models.diffusion.DDPM import DDPM
from WholeGraspPose.models.diffusion.Eps import Eps
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

        self.model = Eps(D=cfg.x_dim).to(self.device)
        if self.cfg.trained_diffusion is not None:
            self.load_model()
        self.diffusion = DDPM().to(self.device)
        self.diffusion.model = self.model

        self.optimizer = torch.optim.AdamW(self.diffusion.model.parameters(), lr=cfg.lr)


    def load_data(self):
        data_dict = torch.load(self.cfg.dataset_path, map_location=self.device)
        return TensorDataset(*[data_dict['mu'], data_dict['var'], data_dict['label']])

    def train_ddpm(self):
        for e in tqdm(range(self.cfg.epoch),desc="Epoch"):
            for _i, (_mu, _var, _label) in enumerate(tqdm(self.train_loader,desc="train loader")):
                self.optimizer.zero_grad()

                x_0 = torch.distributions.Normal(_mu, _var).rsample()
                loss, loss_dict = self.diffusion(x=x_0, condition=_label)
                loss.backward()

                self.writer.add_scalar("loss/diffusion_l2", loss, _i + e *len(self.train_loader) )
                self.writer.flush()

                self.optimizer.step()
            self.save("epoch_" + str(e))

    def load_model(self):
        self.model.load_state_dict(torch.load(self.cfg.trained_diffusion, map_location=self.device), strict=False)

    def dist_metrics(self, p=2):
        sample_batch = 512

        samples = []
        mean_samples = []
        for label in range(3):
            cond_ = torch.nn.functional.one_hot(torch.ones((sample_batch, )).long() * label, 23).float()
            samples_ = self.diffusion.sample(batch_size=sample_batch, ddim=False, condition= cond_)
            samples.append(samples_)
            mean_samples.append(samples_.mean(dim=0)[None, :])

        internal_dist = []
        for samples_l in samples:
            dist = torch.cdist(samples_l, samples_l, p = p)
            mean_internal_dist = torch.sum(dist) / (sample_batch**2 - sample_batch)
            internal_dist.append(mean_internal_dist)
        mean_samples = torch.cat(mean_samples)
        external_dist = torch.cdist(mean_samples, mean_samples, p=2)
        return

    def save(self, ckpt_name):
        torch.save(self.model.state_dict(), "{}/{}.pt".format(self.cfg.save_folder, ckpt_name))

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
    exp_name = "test2"

    cfg = Config(**{
        "exp_name": exp_name,
        "dataset_path":'saga_male_latent_256_label.pt',
        "batch_size": 64,
        "x_dim":256,
        "trained_model":None,
        "epoch":100,
        "save_folder":f"logs/{exp_name}/",
        "lr":5e-5,
        "trained_diffusion": None,
        "cuda_id":0,
    })

    trainer = DiffusionTrainer(cfg)
    trainer.train_ddpm()
    trainer.dist_metrics()
    # trainer.save()
    trainer.original_data()
