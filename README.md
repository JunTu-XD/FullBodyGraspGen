# setup
### Venv
```
mkdir venvs
python3 -m venv venvs/grasp_venv
source venvs/grasp_venv/bin/activate
```
### euler module
```// on euler: module load gcc/8.2.0 python_gpu/3.10.4 open3d/0.9.0 boost/1.74.0 eth_proxy```
```// else:```
```pip install open3d```
```pip install -r requirements.txt```
```
// follow https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
```

### download files
- <strong>Body Models</strong>  
Download [SMPL-X body model and vposer v1.0 model](https://smpl-x.is.tue.mpg.de/index.html) and put them under /body_utils/body_models folder as below:
```
SAGA
│
└───body_utils
    │
    └───body_models 
        │
        └───smplx
        │   └───SMPLX_FEMALE.npz
        │   └───...
        │   
        └───vposer_v1_0
        │   └───snapshots
        │       └───TR00_E096.pt
        │   └───...
        │
        └───VPoser
        │   └───vposerDecoderWeights.npz
        │   └───vposerEnccoderWeights.npz
        │   └───vposerMeanPose.npz
    │
    └───...
│
└───...
```

## Dataset
### 
Download [GRAB](https://grab.is.tue.mpg.de/) object mesh

Download dataset for the first stage (GraspPose) from [[Google Drive]](https://drive.google.com/uc?export=download&id=1OfSGa3Y1QwkbeXUmAhrfeXtF89qvZj54)

Download dataset for the second stage (GraspMotion) from [[Google Drive]](https://drive.google.com/uc?export=download&id=1QiouaqunhxKuv0D0QHv1JHlwVU-F6dWm)

Put them under /dataset as below,
```
SAGA
│
└───dataset 
    │
    └───GraspPose
    │   └───train
    │       └───s1
    │       └───...
    │   └───eval
    │       └───s1
    │       └───...
    │   └───test
    │       └───s1
    │       └───...
    │   
    └───GraspMotion
    │   └───Processed_Traj
    │   └───s1
    │   └───...
    │   
    └───contact_meshes
    │   └───airplane.ply
    │   └───...
│
└───... 
```
### modify source code of SAGA
change 

```x_near, y_near, xidx_near, yidx_near = chd.ChamferDistance(x,y)```

in *train_helper.py* to

```chd.ChamferDistance(x,y)=>chd.ChamferDistance()(x,y)```


## set up on local for visualization
- download SAGA 
- pip install requirements_local.txt
- follow instructions in SAGA repo
- after inference done, download results folder into local code base to use vis_pose.py to show them.

## run the evaluation (fitting+opt+eval)
```
# take 30 different object poses from GRAB test set per object class, and generate 5 random samples per object, test for male only
# can set test object class using --objects (default = ['mug','camera','toothpaste','wineglass','fryingpan','binoculars'])
python eval_grasppose.py --exp_name saga_pretrained_eval --male_pose_ckpt_path pretrained_model/male_grasppose_model.pt --n_object_samples 30 --n_rand_samples_per_object 5 --gender male
```

## compute the eval metrics of one single fitting_results.npz file
```
python eval_metrics.py --exp_name saga_512d_female_eval --gender female --object camera --fitting_path results/saga_512d_female_eval/GraspPose/camera --n_rand_samples_per_object 1
```

