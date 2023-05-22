# setup SAGA
## on euler server
### conda
```conda create -n grasp_conda python=3.8```
### Or venv
```
mkdir venvs
python3 -m venv venvs/grasp_venv
source venvs/grasp_venv/bin/activate
```
### euler module
```module load gcc/8.2.0 python_gpu/3.10.4 open3d/0.9.0 boost/1.74.0 eth_proxy```

```pip install -r requirements.txt```

```
## check python, pytorch+cu version
## modify the py38_cu117_pyt200 to corresponding version
## download https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu117_pyt200/download.html accrodingly
pip install pytorch3d-0.7.3-cp38-cp38-linux_x86_64.whl
```

### download files according to SAGA git.

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

