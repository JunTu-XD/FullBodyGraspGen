import torch

import argparse
import os
import sys
from WholeGraspPose.data.dataloader import LoadData

import numpy as np
# import smplx
import torch

from utils.cfg_parser import Config
from utils.utils import makelogger, makepath
from WholeGraspPose.trainer import Trainer
# from opt_grasppose import *

if __name__ == '__main__':
    
    cwd = os.getcwd()

    best_net = "logs/GraspPose/male_16d_diffusion/checkpoint.pt"

    gender = "male"

    exp_name = "male_16d_diffusion"

    data_path = "./dataset/GraspPose"

    object = "all"

    n_samples =  3 # num of sample objects

    n_rand_samples = 30 # num of random samples generated for each object sample

    vpe_path  = '/configs/verts_per_edge.npy'
    c_weights_path = cwd + '/WholeGraspPose/configs/rhand_weight.npy'
    work_dir = cwd + '/results/{}/GraspPose'.format(exp_name)
    print(work_dir)
    config = {
        'dataset_dir': data_path,
        'work_dir':work_dir,
        'vpe_path': vpe_path,
        'c_weights_path': c_weights_path,
        'exp_name': exp_name,
        'gender': gender,
        'best_net': best_net
    }

    cfg_path = 'WholeGraspPose/configs/WholeGraspPose.yaml'
    cfg = Config(default_cfg_path=cfg_path, **config)

    save_dir = os.path.join(work_dir, object)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)        

    logger = makelogger(makepath(os.path.join(save_dir, '%s.log' % (object)), isfile=True)).info
    
    grabpose = Trainer(cfg=cfg, inference=True, logger=logger)

    # load the training data
    data = LoadData(dataset_dir=data_path, ds_name='train', gender=gender, motion_intent=False, object_class=object)
    rand_test_data = data[:n_samples]
    obj_data = {}
    obj_data['verts_object'] = rand_test_data['verts_object'].permute(0,2,1).to(grabpose.device)
    obj_data['feat_object'] = rand_test_data['feat_object'].permute(0,2,1).to(grabpose.device)
    obj_data['transf_transl'] = rand_test_data['transf_transl'].to(grabpose.device)
    obj_data['markers'] = rand_test_data['markers'].to(grabpose.device)
    obj_data['contacts_object'] = rand_test_data['contacts_object'].view(rand_test_data['contacts_object'].shape[0], 1, -1).to(grabpose.device)
    obj_data['contacts_markers'] = rand_test_data['contacts_markers'].view(rand_test_data['contacts_markers'].shape[0], -1, 1).to(grabpose.device)

    grabpose.full_grasp_net.eval()

    for i in range(n_samples):

        verts_object = obj_data['verts_object'][None, i]
        feat_object = obj_data['feat_object'][None, i]
        transf_transl = obj_data['transf_transl'][None, i]
        contacts_object = obj_data['contacts_object'][None, i]
        markers = obj_data['markers'][None, i]
        contacts_markers = obj_data['contacts_markers'][None, i]

        object_cond = grabpose.full_grasp_net.pointnet(l0_xyz=verts_object, l0_points=feat_object)
        z_encoder = grabpose.full_grasp_net.encode(object_cond, verts_object, feat_object, contacts_object, markers, contacts_markers, transf_transl)
        print("================================================================")
        print(f"The output of the saga encoder for the {i+1}th {object} sample is:")
        print(f"Mean = {z_encoder.loc}")
        print(f"Std = {z_encoder.scale}")

        ## DDIM sampling
        _, _, _, _, l3_xyz, l3_f = object_cond
        _diffusion_params = {"batch_size": n_rand_samples, "condition": None} # if use cond ->"condition": self.diffusion.construct_condition(obj_feature=l3_f, obj_xyz=l3_xyz, transl=transf_transl)
        z_diffusion = grabpose.full_grasp_net.diffusion.sample(ddim=True, **_diffusion_params)
        miu, sigma = torch.mean(z_diffusion, dim = 0), torch.std(z_diffusion, dim = 0)
        print(f"The distribution of the DDIM samples for the {i+1}th {object} sample is:")
        print(f"Mean = {miu}")
        print(f"Std = {sigma}")


