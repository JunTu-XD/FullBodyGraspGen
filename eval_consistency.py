import argparse
import os
import sys
from collections import defaultdict
import json

import numpy as np
# import smplx
import open3d as o3d
import torch
from smplx.lbs import batch_rodrigues
from tqdm import tqdm

from utils.cfg_parser import Config
from utils.utils import makelogger, makepath
from WholeGraspPose.models.fittingop import FittingOP
from WholeGraspPose.models.objectmodel import ObjectModel
from WholeGraspPose.trainer import Trainer

from eval_metrics import evaluate_consistency, evaluate_cross_label_dist, set_torch



#### inference
def load_object_data_random(object_name, n_samples):
    mesh_base = './dataset/contact_meshes'
    obj_mesh_base = o3d.io.read_triangle_mesh(os.path.join(mesh_base, object_name + '.ply'))
    obj_mesh_base.compute_vertex_normals()
    v_temp = torch.FloatTensor(obj_mesh_base.vertices).to(grabpose.device).view(1, -1, 3).repeat(n_samples, 1, 1)
    normal_temp = torch.FloatTensor(obj_mesh_base.vertex_normals).to(grabpose.device).view(1, -1, 3).repeat(n_samples, 1, 1)
    obj_model = ObjectModel(v_temp, normal_temp, n_samples)

    """Prepare transl/global_orient data"""
    """Example: randomly sample object height and orientation"""
    transf_transl_list = torch.rand(n_samples) + 0.6   #### can be customized
    global_orient_list = (np.pi)*torch.rand(n_samples) - np.pi/2   #### can be customized
    transl = torch.zeros(n_samples, 3)   # for object model which is centered at object
    transf_transl = torch.zeros(n_samples, 3)
    transf_transl[:, -1] = transf_transl_list
    global_orient = torch.zeros(n_samples, 3)
    global_orient[:, -1] = global_orient_list
    global_orient_rotmat = batch_rodrigues(global_orient.view(-1, 3)).to(grabpose.device)   # [N, 3, 3]

    object_output = obj_model(global_orient_rotmat, transl.to(grabpose.device), v_temp.to(grabpose.device), normal_temp.to(grabpose.device), rotmat=True)
    object_verts = object_output[0].detach().squeeze().cpu().numpy() if n_samples != 1 else object_output[0].detach().cpu().numpy()
    object_normal = object_output[1].detach().squeeze().cpu().numpy() if n_samples != 1 else object_output[1].detach().cpu().numpy()
    
    index = np.linspace(0, object_verts.shape[1], num=2048, endpoint=False,retstep=True,dtype=int)[0]
    
    verts_object = object_verts[:, index]
    normal_object = object_normal[:, index]
    global_orient_rotmat_6d = global_orient_rotmat.view(-1, 1, 9)[:, :, :6].detach().cpu().numpy()
    feat_object = np.concatenate([normal_object, global_orient_rotmat_6d.repeat(2048, axis=1)], axis=-1)
    
    verts_object = torch.from_numpy(verts_object).to(grabpose.device)
    feat_object = torch.from_numpy(feat_object).to(grabpose.device)
    transf_transl = transf_transl.to(grabpose.device)
    return {'verts_object':verts_object, 'normal_object': normal_object, 'global_orient':global_orient, 'global_orient_rotmat':global_orient_rotmat, 'feat_object':feat_object, 'transf_transl':transf_transl}


def load_object_data_uniform_sample(object_name, n_samples):
    mesh_base = './dataset/contact_meshes'
    obj_mesh_base = o3d.io.read_triangle_mesh(os.path.join(mesh_base, object_name + '.ply'))
    obj_mesh_base.compute_vertex_normals()
    v_temp = torch.FloatTensor(obj_mesh_base.vertices).to(grabpose.device).view(1, -1, 3).repeat(n_samples, 1, 1)
    normal_temp = torch.FloatTensor(obj_mesh_base.vertex_normals).to(grabpose.device).view(1, -1, 3).repeat(n_samples, 1, 1)
    obj_model = ObjectModel(v_temp, normal_temp, n_samples)

    """Prepare transl/global_orient data"""
    """Example: uniformly sample object height and orientation (can be customized)"""
    transf_transl_list = torch.arange(n_samples)*1.0/(n_samples-1) + 0.5
    global_orient_list = (2*np.pi)*torch.arange(n_samples)/n_samples
    # n_samples = transf_transl_list.shape[0] * global_orient_list.shape[0]
    transl = torch.zeros(n_samples, 3)   # for object model which is centered at object
    transf_transl = torch.zeros(n_samples, 3)
    transf_transl[:, -1] = transf_transl_list # .repeat_interleave(global_orient_list.shape[0])
    global_orient = torch.zeros(n_samples, 3)
    global_orient[:, -1] = global_orient_list # .repeat(transf_transl_list.shape[0])  # [6+6+6.....]
    global_orient_rotmat = batch_rodrigues(global_orient.view(-1, 3)).to(grabpose.device)   # [N, 3, 3]

    object_output = obj_model(global_orient_rotmat, transl.to(grabpose.device), v_temp.to(grabpose.device), normal_temp.to(grabpose.device), rotmat=True)
    object_verts = object_output[0].detach().squeeze().cpu().numpy() if n_samples != 1 else object_output[0].detach().cpu().numpy()
    object_normal = object_output[1].detach().squeeze().cpu().numpy() if n_samples != 1 else object_output[1].detach().cpu().numpy()
    
    index = np.linspace(0, object_verts.shape[1], num=2048, endpoint=False,retstep=True,dtype=int)[0]
    
    verts_object = object_verts[:, index]
    normal_object = object_normal[:, index]
    global_orient_rotmat_6d = global_orient_rotmat.view(-1, 1, 9)[:, :, :6].detach().cpu().numpy()
    feat_object = np.concatenate([normal_object, global_orient_rotmat_6d.repeat(2048, axis=1)], axis=-1)
    
    verts_object = torch.from_numpy(verts_object).to(grabpose.device)
    feat_object = torch.from_numpy(feat_object).to(grabpose.device)
    transf_transl = transf_transl.to(grabpose.device)
    return {'verts_object':verts_object, 'normal_object': normal_object, 'global_orient':global_orient, 'global_orient_rotmat':global_orient_rotmat, 'feat_object':feat_object, 'transf_transl':transf_transl}

def inference(grabpose, obj, n_samples, n_rand_samples, object_type, sample_label):
    """ prepare test object data: [verts_object, feat_object(normal + rotmat), transf_transl] """
    ### object centered
    # for obj in grabpose.cfg.object_class:
    if object_type == 'uniform':
        obj_data = load_object_data_uniform_sample(obj, n_samples)
    elif object_type == 'random':
        obj_data = load_object_data_random(obj, n_samples)
    obj_data['feat_object'] = obj_data['feat_object'].permute(0,2,1)
    obj_data['verts_object'] = obj_data['verts_object'].permute(0,2,1)

    n_samples_total = obj_data['feat_object'].shape[0]

    markers_gen = []
    object_contact_gen = []
    markers_contact_gen = []
    for i in range(n_samples_total):
        sample_results = grabpose.full_grasp_net.sample(obj_data['verts_object'][None, i].repeat(n_rand_samples,1,1), 
                                                        obj_data['feat_object'][None, i].repeat(n_rand_samples,1,1), 
                                                        obj_data['transf_transl'][None, i].repeat(n_rand_samples,1),
                                                        label=torch.nn.functional.one_hot(torch.tensor(sample_label), 23).repeat(n_rand_samples,1).float())
        markers_gen.append((sample_results[0]+obj_data['transf_transl'][None, i]))
        markers_contact_gen.append(sample_results[1])
        object_contact_gen.append(sample_results[2])

    markers_gen = torch.cat(markers_gen, dim=0)   # [B, N, 3]
    object_contact_gen = torch.cat(object_contact_gen, dim=0).squeeze()   # [B, 2048]
    markers_contact_gen = torch.cat(markers_contact_gen, dim=0)   # [B, N]

    output = {}
    output['markers_gen'] = markers_gen.detach().cpu().numpy()

    return output

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='grabpose-Testing')

    parser.add_argument('--data_path', default = './dataset/GraspPose', type=str,
                        help='The path to the folder that contains grabpose data')

    parser.add_argument('--objects', default = ['mug','camera','toothpaste','wineglass','fryingpan','binoculars'], type=list, # 6 objects for test (in SAGA paper) ['mug','camera','toothpaste','wineglass','fryingpan','binoculars']
                        help='The list of all the objects for metrics evaluation') 

    parser.add_argument('--config_path', default = None, type=str,
                        help='The path to the confguration of the trained grabpose model')

    parser.add_argument('--exp_name', default = None, type=str,
                        help='experiment name')

    parser.add_argument('--n_object_samples', default = 5, type=int,
                        help='The number of object samples of this object')

    parser.add_argument('--type_object_samples', default = 'random', type=str,
                        help='For the given object mesh, we provide two types of object heights and orientation sampling mode: random / uniform')

    parser.add_argument('--n_rand_samples_per_object', default = 1, type=int,
                        help='The number of whole-body poses random samples generated per object')
    
    parser.add_argument('--pose_ckpt_path', default = 'pretrained_model/male_grasppose_model.pt', type=str,
                        help='checkpoint path for the male model')
    

    parser.add_argument('--diffusion_model_path', default = None, type=str,
                        help='diffusion path')
    
    parser.add_argument('--gender', default = "male", type=str,
                        help='male / female')

    parser.add_argument('--sample_label', default = ['pass', 'inspect', 'offhand', 'use', 'drink', 'pour'], type=list,
                        help='sample label condition') # ['pass', 'inspect', 'offhand', 'use', 'drink', 'pour']
    
    parser.add_argument('--latentD', default = 16, type=int,
                        help='Latent dimension')
    
    args = parser.parse_args()
    n_rand_samples_per_object = args.n_rand_samples_per_object

    cwd = os.getcwd()

    set_torch(deter=True) # guanrantee reproducibility
    

    vpe_path  = '/configs/verts_per_edge.npy'
    c_weights_path = cwd + '/WholeGraspPose/configs/rhand_weight.npy'
    
    body_model_path = cwd + '/body_utils/body_models'
    contact_meshes_path = cwd + '/dataset/contact_meshes'
    diffusion_model_path = os.path.join(cwd, args.diffusion_model_path)
    
    labels = ['call', 'chop', 'clean', 'drink', 'eat', 'fly', 'inspect', 'offhand', 'on', 'open', 'pass', 'peel',
              'play', 'pour', 'screw', 'see', 'set', 'shake', 'stamp', 'staple', 'use', 'wear']
    label_dict = dict()
    for i, l in enumerate(labels):
        label_dict[l] = i

    gender = args.gender
    best_net = os.path.join(cwd, args.pose_ckpt_path)
    work_dir = cwd + '/results/{}/{}/GraspPose'.format(args.exp_name, gender)
    # print(work_dir)
    config = {
        'dataset_dir': args.data_path,
        'work_dir':work_dir,
        'vpe_path': vpe_path,
        'c_weights_path': c_weights_path,
        'exp_name': args.exp_name,
        'gender': gender,
        'best_net': best_net,
        'trained_diffusion': diffusion_model_path,
        'latentD': args.latentD
    }

    cfg_path = 'WholeGraspPose/configs/WholeGraspPose.yaml'
    cfg = Config(default_cfg_path=cfg_path, **config)

    # run the inference for all the listed objects & conditions
    label_consistency = defaultdict(defaultdict)
    cross_label_dist = dict()
    n_labels = len(args.sample_label)
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
    logger = makelogger(makepath(os.path.join(work_dir, 'eval.log'), isfile=True)).info       
    grabpose = Trainer(cfg=cfg, inference=True, logger=logger)
    for object in args.objects:
        markers_gen = []
        for label_cond in args.sample_label:
            print(f"Start evaluating {object} with label {label_cond}")
            label_id = label_dict[label_cond]
            samples_results = inference(grabpose, object, args.n_object_samples, n_rand_samples_per_object, args.type_object_samples, label_id)
            markers_gen.append(samples_results["markers_gen"])
            label_consistency[object][label_cond] = evaluate_consistency(samples_results, n_rand_samples_per_object)

        # compute the cross label distance for each object
        cross_dist_all = []
        for i in range(n_labels-1):
            for j in range(i+1, n_labels):
                cross_dist = evaluate_cross_label_dist(markers_gen[i], markers_gen[j], n_rand_samples_per_object)
                cross_dist_all.append(cross_dist)
        cross_label_dist[object] = np.mean(cross_dist_all)

    output = {
        "label_consistency_score": label_consistency,
        "cross_label_dist": cross_label_dist
    }

    output_path = os.path.join(work_dir,"consistency_eval.json")
    with open(output_path, "w") as outfile:
        json.dump(output, outfile)






    
        


