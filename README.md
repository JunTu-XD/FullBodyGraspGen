# setup
## conda
conda create -n GraspGen python=3.8

## euler module
module load gcc/8.2.0 python_gpu/3.10.4 open3d/0.9.0 boost/1.74.0 eth_proxy

pip install requirements.txt

## check python, pytorch+cu version
## download https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu117_pyt200/download.html accrodingly
pip install pytorch3d-0.7.3-cp38-cp38-linux_x86_64.whl


## download files according to SAGA git.



