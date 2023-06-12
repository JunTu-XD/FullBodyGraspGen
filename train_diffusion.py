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
        for e in range(1):
            for _i, (_mu, _var, _label) in enumerate(tqdm(self.train_loader)):
                self.optimizer.zero_grad()

                x_0 = torch.distributions.Normal(_mu, _var).rsample()
                loss, loss_dict = self.diffusion(x=x_0, condition=_label)
                loss.backward()

                self.writer.add_scalar("loss/diffusion_l2", loss, _i + e *len(self.train_loader) )
                self.writer.flush()

                self.optimizer.step()
    def val(self):
        sample_batch = 256
        cond_1 = torch.nn.functional.one_hot(torch.ones((sample_batch, )).long() * 2, 23).float()
        samples_1 = self.diffusion.sample(batch_size=sample_batch, ddim=False, condition= cond_1)

        cond_0 = torch.nn.functional.one_hot(torch.ones((sample_batch, )).long() * 5, 23).float()
        samples_0 = self.diffusion.sample(batch_size=sample_batch, ddim=False, condition=cond_0)

        return

    def train_imporved_ddpm(self):
        pass

if __name__=="__main__":
    exp_name = "test"

    cfg = Config(**{
        "exp_name": exp_name,
        "dataset_path":'dataset/saga_male_latent_label.pt',
        "batch_size": 64,
        "x_dim":16,
        "trained_model":None,
        "epoch":100,
        "save_folder":f"logs/{exp_name}/",
        "lr":1e-4
    })

    trainer = DiffusionTrainer(cfg)
    trainer.train_ddpm()
    trainer.val()

