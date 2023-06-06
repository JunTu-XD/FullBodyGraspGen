import argparse
import os
import sys

import numpy as np
import open3d as o3d

sys.path.append('.')
sys.path.append('..')
# from WholeGraspPose.data.dataloader_vis import LoadData
from utils.cfg_parser import Config
from visualization_utils import *
import json
import time


def load_single_data(file):
    data = np.load(file, allow_pickle=True)
    ## select object

    obj_name = file.split('/')[-1].split('_')[0]
    mid = data['transf_transl'].shape[0] // 2

    transf_transl = data['transf_transl'][mid, :].reshape(1, -1)
    global_orient_object = data['global_orient_object'][mid, :].reshape(1, -1)

    body_list = {}
    for key in ['transl', 'global_orient', 'body_pose', 'jaw_pose', 'leye_pose', 'reye_pose', 'left_hand_pose',
                'right_hand_pose', 'expression']:
        body_list[key] = 0
    # SMPLX parameters (optional)
    for key in data['body'][()].keys():
        body_list[key] = data['body'][()][key][mid, :].reshape(1, -1)

    output = {}
    output['transf_transl'] = torch.tensor(transf_transl, dtype=torch.float)
    output['global_orient_object'] = torch.tensor(global_orient_object, dtype=torch.float)  # (B, 2048, 3)
    output['obj_name'] = obj_name
    # breakpoint()
    # SMPLX parameters
    output['smplxparams'] = {}
    for key in ['transl', 'global_orient', 'body_pose', 'jaw_pose', 'leye_pose', 'reye_pose', 'left_hand_pose',
                'right_hand_pose', 'expression']:
        output['smplxparams'][key] = torch.tensor(body_list[key], dtype=torch.float)
    return output


def save_pic(geometry, view_status, pic_name, folder_name, data_form, save=True):
    # Create a window to visualize the 3D scene
    visualizer = o3d.visualization.Visualizer()
    if save == True:
        visualizer.create_window()
        for geo in geometry:
            visualizer.add_geometry(geo)
        view_settings = o3d.io.read_pinhole_camera_parameters(view_status)

        # Apply the view status settings to the visualizer
        ctr = visualizer.get_view_control()
        ctr.convert_from_pinhole_camera_parameters(view_settings)

        visualizer.poll_events()
        param = ctr.convert_to_pinhole_camera_parameters()
        o3d.io.write_pinhole_camera_parameters('test.json', param)

        pic_name = './save_pics/{}/{}/{}.png'.format(data_form, folder_name, pic_name)
        visualizer.capture_screen_image(pic_name)

        # Close the visualizer window
        visualizer.destroy_window()
    else:
        # visualize the mesh in an interactive window
        o3d.visualization.draw_geometries(geometry)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='grabpose-Testing')

    # parser.add_argument('--exp_name', default = None, type=str,
    #                     help='experiment name')

    parser.add_argument('--gender', default='male', type=str,
                        help='gender')

    parser.add_argument('--object', default='', type=str,
                        help='object name')

    parser.add_argument('--object_format', default='mesh', type=str,
                        help='pcd or mesh')

    # 'objectmesh' : grasp+object,
    # 'contactmap' : grasp+conatctmap
    parser.add_argument('--visual_cue', default='objectmesh', type=str,
                        help='pcd or mesh')

    parser.add_argument('--ds_format', default='test', type=str,
                        help='pcd or mesh')

    parser.add_argument('--save', default='True', type=str,
                        help='save pics or not')

    parser.add_argument('--file_path', default=None, type=str,
                        help='spath to load and visualize the single file')
    args = parser.parse_args()
    default_cfg_path = './WholeGraspPose/configs/WholeGraspPose.yaml'

    cfg = Config(default_cfg_path=default_cfg_path, \
                 **{
                     'debug': True,
                     "dataset_dir": "./dataset/GraspPose",
                     'gender': args.gender})

    cwd = os.getcwd()

    # ground
    x_range = np.arange(-5, 50, 1)
    y_range = np.arange(-5, 50, 1)
    z_range = np.arange(0, 1, 1)
    gp_lines, gp_pcd = create_lineset(x_range, y_range, z_range)
    gp_lines.paint_uniform_color(color_hex2rgb('#bdbfbe'))  # grey
    gp_pcd.paint_uniform_color(color_hex2rgb('#bdbfbe'))  # grey
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.25)

    if args.file_path is not None:
        data = load_single_data(args.file_path)

        body_model_path = './body_utils/body_models'
        contact_meshes_path = './dataset/contact_meshes'
        gender = args.gender
        transl = np.zeros_like(data['transf_transl'])

        object_mesh = get_object_mesh(contact_meshes_path, data['obj_name'], 'GRAB', transl,
                                      data['global_orient_object'], n_samples=1)

        body_mesh, _ = get_body_mesh(body_model_path, data['smplxparams'], gender, n_samples=1)

        visualization_list = [body_mesh[0], object_mesh[0], coord, gp_lines, gp_pcd]
        o3d.visualization.draw_geometries(visualization_list)


    else:

        load_data = LoadData(dataset_dir=cfg.dataset_dir, ds_name=args.ds_format, gender=cfg.gender,
                             motion_intent=cfg.motion_intent,
                             object_class=cfg.object_class, debug=False)
        bodies = {}
        for k, v in load_data.ds['smplxparams'].items():
            bodies[k] = v.detach().numpy()

        objs = {}
        objs['transl'] = load_data.ds['transf_transl'].detach().numpy()
        objs['global_orient'] = load_data.ds['global_orient_object'].detach().numpy()

        objs['transl'] = np.zeros_like(objs['transl'])

        body_model_path = './body_utils/body_models'
        contact_meshes_path = './dataset/contact_meshes'

        gender = args.gender
        object_name = args.object
        name_list = load_data.ds['obj_name']
        save_name_list = load_data.ds['save_name']
        save_folder_list = load_data.ds['save_folder']

        n_samples = len(name_list)

        # Prepare mesh and pcd
        # no need for visualization of obj pcd
        # object_pcd = object_pcd = get_pcd(data['object'][()]['verts_object'][:n_samples], data['contact'][()]['object'][:n_samples])  # together with the contact map info
        object_mesh_list = []
        # breakpoint()
        for i in range(len(name_list)):
            object_mesh = get_object_mesh(contact_meshes_path, name_list[i], 'GRAB', objs['transl'][i],
                                          objs['global_orient'][i], n_samples=1)
            object_mesh_list.append(object_mesh[0])
        body_mesh, _ = get_body_mesh(body_model_path, bodies, gender, n_samples)
        # breakpoint()

        # location of the view settings file
        view_file = './visualization/view_status_rhs.json'
        view_file_val_female = './visualization/view_status_val_female.json'
        # view_status =		{
        # 		"boundingbox_max" : np.array([ 0.36077451705932617, 0.7790188193321228, 0.80461704730987549 ]),
        # 		"boundingbox_min" : np.array([ -0.21544934809207916, -0.035889849066734314, -0.96224409341812134 ]),
        # 		"field_of_view" : 60.0,
        # 		"front" : np.array([ 0.79444279499857984, -0.47723617991951023, 0.37564115063538689 ]),
        # 		"lookat" : np.array([ -0.074169491056332013, 0.22926275114816022, -0.019146877259963758 ]),
        # 		"up" : np.array([ -0.33119004936555163, 0.17803263919218634, 0.92661617220050096 ]),
        # 		"zoom" : 0.6399999999999999
        # 	}

        if args.visual_cue == "objectmesh":
            # draw grasp pose + object
            for i in range(n_samples):
                print(body_mesh[i])
                visualization_list = [body_mesh[i], object_mesh_list[i], coord, gp_lines, gp_pcd]

                # o3d.visualization.draw_geometries(visualization_list)
                save_pic(visualization_list, view_file_val_female, save_name_list[i], save_folder_list[i],
                         args.ds_format, save=True)
        else:
            # draw grasp pose + contact map
            for i in range(n_samples):
                visualization_list = [body_mesh[i], object_pcd[i], coord, gp_lines, gp_pcd]
                o3d.visualization.draw_geometries(visualization_list)