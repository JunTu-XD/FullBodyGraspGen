# setup SAGA
## on euler server
### conda
```conda create -n grasp_conda python=3.8```

### euler module
```module load gcc/8.2.0 python_gpu/3.10.4 open3d/0.9.0 boost/1.74.0 eth_proxy```

```pip install requirements.txt```

```
## check python, pytorch+cu version
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
