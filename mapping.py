import argparse
import os
import torch
import torch.optim as optim
import torch.nn as nn
from datetime import datetime

import torch.utils.data as td
import time
from torch.utils.tensorboard import SummaryWriter

from WholeGraspPose.trainer import Trainer

from utils.cfg_parser import Config

def save_dataloader(grabpose):
    
    torch.autograd.set_detect_anomaly(True)
    grabpose.full_grasp_net.eval()   

    starttime = datetime.now().replace(microsecond=0)

    ep_start_time = datetime.now().replace(microsecond=0)

    mu_set = []
    var_set = []
    label_set = []
    for it, dorig in enumerate(grabpose.ds_train):
        
        dorig = {k: dorig[k].to(grabpose.device) for k in dorig.keys() if k!='smplxparams'}  
        dorig['verts_object'] = dorig['verts_object'].permute(0,2,1)
        dorig['feat_object'] = dorig['feat_object'].permute(0,2,1)
        dorig['contacts_object'] = dorig['contacts_object'].view(dorig['contacts_object'].shape[0], 1, -1)
        dorig['contacts_markers'] = dorig['contacts_markers'].view(dorig['contacts_markers'].shape[0], -1, 1)
        
        with torch.no_grad():
            mu, var, label = grabpose.full_grasp_net(**dorig)

        mu_set.append(mu)
        var_set.append(var)
        label_set.append(label)
        # if it == 2:
        #     break
        

    mu_set    = torch.cat(mu_set, dim=0)
    var_set   = torch.cat(var_set, dim=0)
    label_set = torch.cat(label_set, dim=0)

    
    save_data_dict ={'mu': mu_set,'var':var_set,
                     'label': label_set}
    
    torch.save(save_data_dict,'./saga_male_latent_512_label.pt')
    

    endtime = datetime.now().replace(microsecond=0)

       
    return 
    
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GrabNet-Training')

    parser.add_argument('--work-dir', default='logs/GraspPose', type=str,
                        help='The path to the downloaded grab data')

    parser.add_argument('--gender', default=None, type=str,
                        help='The gender of dataset')

    parser.add_argument('--data_path', default = '/cluster/work/cvl/wuyan/data/GRAB-series/GrabPose_r_fullbody/data', type=str,
                        help='The path to the folder that contains grabpose data')

    parser.add_argument('--batch-size', default=64, type=int,
                        help='Training batch size')

    parser.add_argument('--n-workers', default=8, type=int,
                        help='Number of PyTorch dataloader workers')

    parser.add_argument('--lr', default=5e-4, type=float,
                        help='Training learning rate')

    parser.add_argument('--kl-coef', default=0.5, type=float,
                        help='KL divergence coefficent for Coarsenet training')

    parser.add_argument('--use-multigpu', default=False,
                        type=lambda arg: arg.lower() in ['true', '1'],
                        help='If to use multiple GPUs for training')

    parser.add_argument('--exp_name', default = None, type=str,
                        help='experiment name')

    parser.add_argument('--pose_ckpt_path', default = None, type=str,
                        help='checkpoint path')
    
    args = parser.parse_args()

   
   

    cwd = os.getcwd()

   

    cfg = {
        'batch_size': args.batch_size,
        'n_workers': args.n_workers,
        'use_multigpu': args.use_multigpu,
        'kl_coef': args.kl_coef,
        'dataset_dir': args.data_path,
        'base_dir': cwd,
        'work_dir': "./",
        'base_lr': args.lr,
        'gender': args.gender,
        'exp_name': args.exp_name,
        'debug':False
    }
    cfg_path = 'WholeGraspPose/configs/WholeGraspPose.yaml'
    cfg = Config(default_cfg_path=cfg_path, **cfg)

    grabpose = Trainer(cfg=cfg, inference = False)

    save_dataloader(grabpose)

