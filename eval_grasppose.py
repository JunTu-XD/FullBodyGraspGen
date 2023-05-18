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

from eval_metrics import evaluate, set_torch

from WholeGraspPose.data.dataloader import LoadData

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
    n_samples = transf_transl_list.shape[0] * global_orient_list.shape[0]
    transl = torch.zeros(n_samples, 3)   # for object model which is centered at object
    transf_transl = torch.zeros(n_samples, 3)
    transf_transl[:, -1] = transf_transl_list.repeat_interleave(global_orient_list.shape[0])
    global_orient = torch.zeros(n_samples, 3)
    global_orient[:, -1] = global_orient_list.repeat(transf_transl_list.shape[0])  # [6+6+6.....]
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

def load_object_data_from_test_set(object_name, n_object_samples, gender_class):
    data_path = './dataset/GraspPose'
    ds_test = LoadData(dataset_dir=data_path, ds_name='test', gender=gender_class, motion_intent=False, object_class=object_name)
    rnd_frames = np.linspace(0, len(ds_test)-1, n_object_samples)
    rand_test_data = ds_test[rnd_frames]
    obj_data = {}
    obj_data['verts_object'] = rand_test_data['verts_object'].to(grabpose.device)
    obj_data['feat_object'] = rand_test_data['feat_object'].to(grabpose.device)
    obj_data['transf_transl'] = rand_test_data['transf_transl'].to(grabpose.device)
    obj_data['global_orient'] = rand_test_data['global_orient_object']
    obj_data['global_orient_rotmat'] = batch_rodrigues(obj_data['global_orient'].view(-1, 3)).to(grabpose.device)
    obj_data['normal_object'] = rand_test_data['normal_object']
    return obj_data

def inference(grabpose, obj, n_samples, n_rand_samples, object_type, save_dir):
    """ prepare test object data: [verts_object, feat_object(normal + rotmat), transf_transl] """
    ### object centered
    # for obj in grabpose.cfg.object_class:
    if object_type == 'uniform':
        obj_data = load_object_data_uniform_sample(obj, n_samples)
    elif object_type == 'random':
        obj_data = load_object_data_random(obj, n_samples)
    elif object_type == 'testset_random':
        obj_data = load_object_data_from_test_set(obj, n_samples, grabpose.cfg.gender)
    obj_data['feat_object'] = obj_data['feat_object'].permute(0,2,1)
    obj_data['verts_object'] = obj_data['verts_object'].permute(0,2,1)

    n_samples_total = obj_data['feat_object'].shape[0]

    markers_gen = []
    object_contact_gen = []
    markers_contact_gen = []
    for i in range(n_samples_total):
        sample_results = grabpose.full_grasp_net.sample(obj_data['verts_object'][None, i].repeat(n_rand_samples,1,1), obj_data['feat_object'][None, i].repeat(n_rand_samples,1,1), obj_data['transf_transl'][None, i].repeat(n_rand_samples,1))
        markers_gen.append((sample_results[0]+obj_data['transf_transl'][None, i]))
        markers_contact_gen.append(sample_results[1])
        object_contact_gen.append(sample_results[2])

    markers_gen = torch.cat(markers_gen, dim=0)   # [B, N, 3]
    object_contact_gen = torch.cat(object_contact_gen, dim=0).squeeze()   # [B, 2048]
    markers_contact_gen = torch.cat(markers_contact_gen, dim=0)   # [B, N]

    output = {}
    output['markers_gen'] = markers_gen.detach().cpu().numpy()
    output['markers_contact_gen'] = markers_contact_gen.detach().cpu().numpy()
    output['object_contact_gen'] = object_contact_gen.detach().cpu().numpy()
    output['normal_object'] = obj_data['normal_object']#.repeat(n_rand_samples, axis=0)
    output['transf_transl'] = obj_data['transf_transl'].detach().cpu().numpy()#.repeat(n_rand_samples, axis=0)
    output['global_orient_object'] = obj_data['global_orient'].detach().cpu().numpy()#.repeat(n_rand_samples, axis=0)
    output['global_orient_object_rotmat'] = obj_data['global_orient_rotmat'].detach().cpu().numpy()#.repeat(n_rand_samples, axis=0)
    output['verts_object'] = (obj_data['verts_object']+obj_data['transf_transl'].view(-1,3,1).repeat(1,1,2048)).permute(0, 2, 1).detach().cpu().numpy()#.repeat(n_rand_samples, axis=0)

    save_path = os.path.join(save_dir, 'markers_results.npy')
    np.save(save_path, output)
    print('Saving to {}'.format(save_path))

    return output

def fitting_data_save(save_data,
              markers,
              markers_fit,
              smplxparams,
              gender,
              object_contact, body_contact,
              object_name, verts_object, global_orient_object, transf_transl_object):
    # markers & markers_fit
    save_data['markers'].append(markers)
    save_data['markers_fit'].append(markers_fit)
    # print('markers:', markers.shape)

    # body params
    for key in save_data['body'].keys():
        # print(key, smplxparams[key].shape)
        save_data['body'][key].append(smplxparams[key].detach().cpu().numpy())
    # object name & object params
    save_data['object_name'].append(object_name)
    save_data['gender'].append(gender)
    save_data['object']['transl'].append(transf_transl_object)
    save_data['object']['global_orient'].append(global_orient_object)
    save_data['object']['verts_object'].append(verts_object)

    # contact
    save_data['contact']['body'].append(body_contact)
    save_data['contact']['object'].append(object_contact)

#### fitting

def pose_opt(grabpose, samples_results, n_random_samples, obj, gender, save_dir, logger, device):
    # prepare objects
    n_samples = len(samples_results['verts_object'])
    verts_object = torch.tensor(samples_results['verts_object'])[:n_samples].to(device)  # (n, 2048, 3)
    normals_object = torch.tensor(samples_results['normal_object'])[:n_samples].to(device)  # (n, 2048, 3)
    global_orients_object = torch.tensor(samples_results['global_orient_object'])[:n_samples].to(device)  # (n, 2048, 3)
    transf_transl_object = torch.tensor(samples_results['transf_transl'])[:n_samples].to(device)  # (n, 2048, 3)

    # prepare body markers
    markers_gen = torch.tensor(samples_results['markers_gen']).to(device)  # (n*k, 143, 3)
    object_contacts_gen = torch.tensor(samples_results['object_contact_gen']).view(markers_gen.shape[0], -1, 1).to(device)  #  (n, 2048, 1)
    markers_contacts_gen = torch.tensor(samples_results['markers_contact_gen']).view(markers_gen.shape[0], -1, 1).to(device)   #  (n, 143, 1)

    print('Fitting {} {} samples for {}...'.format(n_samples, cfg.gender, obj.upper()))

    fittingconfig={ 'init_lr_h': 0.008,
                            'num_iter': [300,400,500],
                            'batch_size': 1,
                            'num_markers': 143,
                            'device': device,
                            'cfg': cfg,
                            'verbose': False,
                            'hand_ncomps': 24,
                            'only_rec': False,     # True / False 
                            'contact_loss': 'contact',  # contact / prior / False
                            'logger': logger,
                            'data_type': 'markers_143',
                            }
    fittingop = FittingOP(fittingconfig)

    save_data_gen = {}
    for data in [save_data_gen]:
        data['markers'] = []
        data['markers_fit'] = []
        data['body'] = {}
        for key in ['betas', 'transl', 'global_orient', 'body_pose', 'leye_pose', 'reye_pose', 'left_hand_pose', 'right_hand_pose']:
            data['body'][key] = []
        data['object'] = {}
        for key in ['transl', 'global_orient', 'verts_object']:
            data['object'][key] = []
        data['contact'] = {}
        for key in ['body', 'object']:
            data['contact'][key] = []
        data['gender'] = []
        data['object_name'] = []


    for i in tqdm(range(n_samples)):
        # prepare object 
        vert_object = verts_object[None, i, :, :]
        normal_object = normals_object[None, i, :, :]

        marker_gen = markers_gen[i*n_random_samples:(i+1)*n_random_samples, :, :]
        object_contact_gen = object_contacts_gen[i*n_random_samples:(i+1)*n_random_samples, :, :]
        markers_contact_gen = markers_contacts_gen[i*n_random_samples:(i+1)*n_random_samples, :, :]

        for k in range(n_random_samples):
            print('Fitting for {}-th GEN...'.format(k+1))
            markers_fit_gen, smplxparams_gen, loss_gen = fittingop.fitting(marker_gen[None, k, :], object_contact_gen[None, k, :], markers_contact_gen[None, k], vert_object, normal_object, gender)
            fitting_data_save(save_data_gen,
                    marker_gen[k, :].detach().cpu().numpy().reshape(1, -1 ,3),
                    markers_fit_gen[-1].squeeze().reshape(1, -1 ,3),
                    smplxparams_gen[-1],
                    gender,
                    object_contact_gen[k].detach().cpu().numpy().reshape(1, -1), markers_contact_gen[k].detach().cpu().numpy().reshape(1, -1),
                    obj, vert_object.detach().cpu().numpy(), global_orients_object[i].detach().cpu().numpy(), transf_transl_object[i].detach().cpu().numpy())


    for data in [save_data_gen]:
        # for data in [save_data_gt, save_data_rec, save_data_gen]:
            data['markers'] = np.vstack(data['markers'])  
            data['markers_fit'] = np.vstack(data['markers_fit'])
            for key in ['betas', 'transl', 'global_orient', 'body_pose', 'leye_pose', 'reye_pose', 'left_hand_pose', 'right_hand_pose']:
                data['body'][key] = np.vstack(data['body'][key])
            for key in ['transl', 'global_orient', 'verts_object']:
                data['object'][key] = np.vstack(data['object'][key])
            for key in ['body', 'object']:
                data['contact'][key] = np.vstack(data['contact'][key])

    np.savez(os.path.join(save_dir, 'fitting_results.npz'), **save_data_gen)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='grabpose-Testing')

    parser.add_argument('--data_path', default = './dataset/GraspPose', type=str,
                        help='The path to the folder that contains grabpose data')

    parser.add_argument('--objects', default = ['mug','camera','toothpaste','wineglass','fryingpan','binoculars'], type=list, # 6 objects for test (in SAGA paper)
                        help='The list of all the objects for metrics evaluation') 

    parser.add_argument('--config_path', default = None, type=str,
                        help='The path to the confguration of the trained grabpose model')

    parser.add_argument('--exp_name', default = None, type=str,
                        help='experiment name')

    parser.add_argument('--pose_ckpt_folder', default = None, type=str,
                        help='checkpoint folder')

    parser.add_argument('--n_object_samples', default = 5, type=int,
                        help='The number of object samples of this object')

    parser.add_argument('--type_object_samples', default = 'random', type=str,
                        help='For the given object mesh, we provide two types of object heights and orientation sampling mode: random / uniform / testset_random')

    parser.add_argument('--n_rand_samples_per_object', default = 1, type=int,
                        help='The number of whole-body poses random samples generated per object')

    parser.add_argument('--gender', default = "male", type=str,
                        help='male, female, all')
    
    args = parser.parse_args()
    n_rand_samples_per_object = args.n_rand_samples_per_object

    cwd = os.getcwd()

    set_torch(deter=True) # guanrantee reproducibility
    

    vpe_path  = '/configs/verts_per_edge.npy'
    c_weights_path = cwd + '/WholeGraspPose/configs/rhand_weight.npy'
    
    body_model_path = cwd + '/body_utils/body_models'
    contact_meshes_path = cwd + '/dataset/contact_meshes'
    
    # the four metrics for evaluation
    sum_contact, sum_apd, sum_inter_vol, sum_inter_depth = 0., 0., 0., 0.
    n_samples = 0
    n_groups_samples = 0

    # test for the specified gender type
    if args.gender == "male":
        genders = ["male"]
    elif args.gender == "female":
        genders = ["female"]
    else:
        genders = ['male', 'female']

    for gender in genders:
        model_name = gender + "_grasppose_model.pt"
        best_net = os.path.join(cwd, args.pose_ckpt_folder, model_name)
        work_dir = cwd + '/results/{}/{}/GraspPose'.format(args.exp_name, gender)
        # print(work_dir)
        config = {
            'dataset_dir': args.data_path,
            'work_dir':work_dir,
            'vpe_path': vpe_path,
            'c_weights_path': c_weights_path,
            'exp_name': args.exp_name,
            'gender': gender,
            'best_net': best_net
        }

        cfg_path = 'WholeGraspPose/configs/WholeGraspPose.yaml'
        cfg = Config(default_cfg_path=cfg_path, **config)

        # run the inference for all the listed objects
        for object in args.objects:
            save_dir = os.path.join(work_dir, object)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)        

            logger = makelogger(makepath(os.path.join(save_dir, '%s.log' % (object)), isfile=True)).info
            
            grabpose = Trainer(cfg=cfg, inference=True, logger=logger)
            
            samples_results = inference(grabpose, object, args.n_object_samples, n_rand_samples_per_object, args.type_object_samples, save_dir)
            fitting_results = pose_opt(grabpose, samples_results, n_rand_samples_per_object, object, cfg.gender, save_dir, logger, grabpose.device)
        
            # evaluate the results with 4 metrics
            eval_i = evaluate(body_model_path, contact_meshes_path, save_dir, gender, object, n_rand_samples_per_object)
            n_samples += len(eval_i['contact'])
            sum_contact += np.sum(eval_i['contact'])
            n_groups_samples += len(eval_i['apd'])
            sum_apd += np.sum(eval_i['apd'])
            sum_inter_vol += np.sum(eval_i['inter_vol'])
            sum_inter_depth += np.sum(eval_i['inter_depth'])
    
    output = dict()
    output['apd'] = sum_apd / n_groups_samples
    output['inter_vol'] = sum_inter_vol / n_samples
    output['inter_depth'] = sum_inter_depth / n_samples
    output['contact_ratio'] = sum_contact / n_samples
    output['n_samples'] = n_samples
    output['n_rand_samples_per_object'] = n_rand_samples_per_object
    output['n_object_samples'] = args.n_object_samples
    print("=================================================")
    print("Final evaluation results for experiemnt:".format(args.exp_name))
    print("APD: {}".format(output['apd']))
    print("AVG_inter_vol: {}".format(output['inter_vol']))
    print("AVG_inter_depth: {}".format(output['inter_depth']))
    print("contact_ratio: {}".format(output['contact_ratio']))
    print("=================================================")
    # write to the json file
    output_path = os.path.join(cwd + '/results/' + args.exp_name, "eval_all.json")
    with open(output_path, "w") as outfile:
        json.dump(output, outfile)
    print("The final evaluation results have been written to {}".format(output_path))
 