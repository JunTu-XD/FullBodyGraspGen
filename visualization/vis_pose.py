import argparse
import os
import sys

import numpy as np
import open3d as o3d
import open3d.visualization as ov

sys.path.append('../')
from WholeGraspPose.data.dataloader import LoadData
from utils.cfg_parser import Config
from visualization_utils import *

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='grabpose-Testing')

    parser.add_argument('--exp_name', default=None, type=str,
                        help='experiment name')

    parser.add_argument('--gender', default=None, type=str,
                        help='gender')

    parser.add_argument('--object', default=None, type=str,
                        help='object name')

    parser.add_argument('--object_format', default='mesh', type=str,
                        help='pcd or mesh')

    # 'objectmesh' : grasp+object, 
    # 'contactmap' : grasp+conatctmap
    parser.add_argument('--visual_cue', default='objectmesh', type=str,
                        help='pcd or mesh')

    args = parser.parse_args()

    default_cfg_path = 'WholeGraspPose/configs/WholeGraspPose.yaml'

    cfg = Config(default_cfg_path=default_cfg_path, **{
        'debug': True,
        "dataset_dir": "./dataset/GraspPose",
        'gender': "female"
    })

    cwd = os.getcwd()

    load_data = LoadData(dataset_dir=cfg.dataset_dir, ds_name='test', gender=cfg.gender,
                         motion_intent=cfg.motion_intent,
                         object_class=cfg.object_class, debug=cfg.debug)

    load_path = 'results/{}/GraspPose/{}/fitting_results.npz'.format(args.exp_name, args.object)
    body_model_path = 'body_utils/body_models'
    contact_meshes_path = 'dataset/contact_meshes'

    # data = np.load(load_path, allow_pickle=True)
    gender = args.gender
    object_name = "mug"

    n_samples = 10
    st = 200
    bodies = {}
    for k, v in load_data.ds['smplxparams'].items():
        bodies[k] = v[st:st + n_samples, :]

    objs = {}
    objs['transl'] = load_data.ds['transf_transl'][st:st + n_samples].detach().numpy()
    objs['global_orient'] = load_data.ds['global_orient_object'][st:st + n_samples].detach().numpy()
    objs['verts_object'] = load_data.ds['verts_object'][st:st + n_samples].detach().numpy()
    objs['contacts_object'] = load_data.ds['contacts_object'][st:st + n_samples].detach().numpy()
    objs['transl'][:, :] = 0

    # Prepare mesh and pcd

    object_pcd = get_pcd(objs['verts_object'][:n_samples],
                         objs['contacts_object'][:n_samples])  # together with the contact map info
    object_mesh = get_object_mesh(contact_meshes_path, object_name, 'GRAB', objs['transl'][:n_samples],
                                  objs['global_orient'][:n_samples], n_samples)
    body_mesh, _ = get_body_mesh(body_model_path, bodies, gender, n_samples)

    # ground
    x_range = np.arange(-5, 50, 1)
    y_range = np.arange(-5, 50, 1)
    z_range = np.arange(0, 1, 1)
    gp_lines, gp_pcd = create_lineset(x_range, y_range, z_range)
    gp_lines.paint_uniform_color(color_hex2rgb('#bdbfbe'))  # grey
    gp_pcd.paint_uniform_color(color_hex2rgb('#bdbfbe'))  # grey
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.25)

    if args.visual_cue == "objectmesh":
        # draw grasp pose + object
        for i in range(n_samples):
            print(body_mesh[i])
            visualization_list = [body_mesh[i], object_mesh[i], coord, gp_lines, gp_pcd]
            lookat = np.array([-0.41887315176225104, 1.5128665737144875, -0.35080178041665006])[:,None]
            front = np.array([])[:,None]
            up = np.array([])[:, None]

            o3d.visualization.draw_geometries(visualization_list)
            vis = o3d.visualization.Visualizer()
            ctrl = vis.get_view_control()
    else:
        # draw grasp pose + contact map
        for i in range(n_samples):
            visualization_list = [body_mesh[i], object_pcd[i], coord, gp_lines, gp_pcd]
            o3d.visualization.draw_geometries(visualization_list)
