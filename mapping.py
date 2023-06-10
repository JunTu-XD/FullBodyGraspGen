# import argparse
# import os
# #import sys
#from collections import defaultdict

# import numpy as np
# import smplx
import argparse
import os

import torch
# from smplx.lbs import batch_rodrigues
#from tqdm import tqdm

#from utils.cfg_parser import Config
#from utils.utils import makelogger, makepath
# from WholeGraspPose.models.fittingop import FittingOP
# from WholeGraspPose.models.objectmodel import ObjectModel
# from WholeGraspPose.trainer import Trainer


# from pdb import set_trace as debug
import torch.optim as optim
import torch.nn as nn
from datetime import datetime

import torch.utils.data as td
import time
from torch.utils.tensorboard import SummaryWriter

from trainer import Trainer
from utils.cfg_parser import Config


class MapperNet(torch.nn.Module):
    def __init__(self):
        super(MapperNet,self).__init__()
        self.fc1 = nn.Linear(512,256)
        self.BN1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256,128)
        self.BN2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128,16)
 
        #self.out_activate =    #TODO
        self.gelu = nn.GELU()
        self.dropout =nn.Dropout1d(0.2)
        

    def forward(self,X):
        out1 = self.gelu(self.BN1(self.fc1(X)))
        out2 = self.gelu(self.BN2(self.fc2(out1)))
        out3 = self.dropout(self.fc3(out2))
        
        return out3
    
    
class MapperNet_female(torch.nn.Module):
    def __init__(self):
        super(MapperNet_female,self).__init__()
        self.fc1 = nn.Linear(512,256)
        self.BN1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256,128)
        self.BN2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128,64)
        self.BN3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64,16)
        
 
        #self.out_activate =    #TODO
        self.gelu = nn.GELU()
        self.dropout =nn.Dropout1d(0.2)
        

    def forward(self,X):
        out1 = self.gelu(self.BN1(self.fc1(X)))
        out2 = self.gelu(self.BN2(self.fc2(out1)))
        out3 = self.gelu(self.BN3(self.fc3(out2)))
        out4 = self.dropout(self.fc4(out3))
        
        return out4
         



class Mapper:
    def __init__(self):
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = MapperNet().to(self.device)
        self.vars_net = [var[1] for var in self.model.named_parameters()]
        self.optimizer_net = optim.Adam(self.vars_net, lr=5e-4, weight_decay=0.0005) 
        self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer_net, milestones=[20,40,60], gamma=0.5)
        
    
    
def train_mapper(mapper, dir=''):
    #load the dataset
    
    writer = SummaryWriter()
    data_dict = torch.load(dir)
    names = list(data_dict.keys())
    train_set = td.TensorDataset(data_dict[names[0]],data_dict[names[1]])
    eval_set = td.TensorDataset(data_dict[names[2]],data_dict[names[3]])
    
    batch_s = 128
    train_loader = td.DataLoader(train_set,shuffle=True,drop_last=True,batch_size=batch_s)
    eval_loader = td.DataLoader(eval_set,shuffle=False,drop_last=False,batch_size=batch_s)
    
    
    
    n_epochs = 100
    #log out the initial cfg
    print("Total num of epochs is {:.0f}".format(n_epochs))
    print("Batch size of the training loader is {:.0f}".format(batch_s))
    torch.autograd.set_detect_anomaly(True)
   
    #Dropout layer not working
    #TODO epoch timing
    starttime = time.time()
    
    
    for epoch_num in range(n_epochs):
        print("Now the epoch number is {:+.0f}".format(epoch_num))
        ep_start_time = time.time()
        
        loss = 0
        loss_test=[]
        train_loss_mu = 0
        eval_loss_mu =[]
        mapper.model.train()
        
        for i, (input,label) in enumerate(train_loader):
            t_label = label[:,:16]
            t_mu    = label[:,16:32]
            #t_var   = label[32:48,:]
   
            
            mapper.optimizer_net.zero_grad()
           
            pred = mapper.model(input)
            losser = nn.MSELoss()
            loss = losser(pred,t_label)
            loss.backward()
            mapper.optimizer_net.step()
            mapper.lr_scheduler.step()
            
            with torch.no_grad():
                mu_losser = nn.MSELoss()
                train_loss_mu   = mu_losser(pred,t_mu)
        writer.add_scalar('Loss/train',float(loss),epoch_num)
        writer.add_scalar('Mu_bias/train',float(train_loss_mu),epoch_num)
        
            
        
        ####################################evalation#################################
        mapper.model.eval()

        with torch.no_grad():
            for i, (input,label) in enumerate(eval_loader):
                e_label = label[:,:16]
                e_mu    = label[:,16:32]
                pred =mapper.model(input)
    

             
                loss_eval =nn.MSELoss()
                loss_mu =nn.MSELoss()
                loss_test.append(loss_eval(pred,e_label))
                eval_loss_mu.append(loss_mu(pred,e_mu))
            loss_eval = torch.mean(torch.Tensor(loss_test))
            loss_mu = torch.mean(torch.Tensor(eval_loss_mu))
        writer.add_scalar('Loss/eval',float(loss_eval),epoch_num)
        writer.add_scalar('Mu_bias/eval',float(loss_mu),epoch_num)
        
        


        ep_end_time = time.time()
       
        time_elapsed =  ep_end_time-ep_start_time            
        print("Epoch {:.0f}: The training time is {:.0f}m {:.0f}".format(epoch_num,time_elapsed//60,time_elapsed%60))
        # print("Epoch {:.0f}: The training loss is {:+.5f}".format(epoch_num,loss))
        # print("Epoch {:.0f}: The testing loss is {:+.5f}".format(epoch_num,loss_eval))
        print("---------------------------------------------------------------------------------------------")
    endtime = datetime.now().replace(microsecond=0)
    #total_time = endtime-starttime
    torch.save(mapper.model,'female-mapper_net.pkl')
    torch.save(mapper.model.state_dict(),'female-mapper_net_params.pkl')
    
    #print("whole time is {:.0f}m {:.0f}".format(total_time // 60, total_time % 60))
       
    return #, train_loss_dict_rnet

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
    
    torch.save(save_data_dict,'./saga_male_latent_label.pt')
    

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

   
    work_dir = os.path.join(args.work_dir, args.exp_name)

    cwd = os.getcwd()

    best_net = os.path.join(cwd, args.pose_ckpt_path)
   

    cfg = {
        'batch_size': args.batch_size,
        'n_workers': args.n_workers,
        'use_multigpu': args.use_multigpu,
        'kl_coef': args.kl_coef,
        'dataset_dir': args.data_path,
        'base_dir': cwd,
        'work_dir': work_dir,
        'base_lr': args.lr,
        'best_net': best_net,
        'gender': args.gender,
        'exp_name': args.exp_name,
    }
    cfg_path = 'WholeGraspPose/configs/WholeGraspPose.yaml'
    cfg = Config(default_cfg_path=cfg_path, **cfg)



    grabpose = Trainer(cfg=cfg, inference = False)

    save_dataloader(grabpose)
    # torch.cuda.empty_cache()
    # mapper = Mapper()
    # train_mapper(mapper,'./female_data_dict.pt')
