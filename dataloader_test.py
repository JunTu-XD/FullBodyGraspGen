import os

from WholeGraspPose.data.dataloader import LoadData
from utils.cfg_parser import Config


def test_loader():
    cwd = os.getcwd()
    default_cfg_path = 'WholeGraspPose/configs/WholeGraspPose.yaml'

    cfg = {
        'batch_size': 64,
        'n_workers': 2,
        'use_multigpu': False,
        'kl_coef': 0.5,
        'dataset_dir': "./dataset/GraspPose/",
        'base_dir': cwd,
        'work_dir': "./",
        'base_lr': 5e-4,
        'best_net': None,
        'gender': 'female',
        'exp_name': "test_exp",
        'debug': True
    }

    cfg = Config(default_cfg_path=default_cfg_path, **cfg)

    ds_name = 'val'
    ds_val = LoadData(dataset_dir=cfg.dataset_dir, ds_name=ds_name, gender=cfg.gender, debug=cfg.debug)