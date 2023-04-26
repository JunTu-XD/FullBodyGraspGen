# FullBodyGraspGen

## conda
install conda in /cluster/scratch/{username}/
```
module load gcc/8.2.0 python_gpu/3.10.4 boost/1.62.0 
conda create -n grasp python=3.7.11

conda activate grasp
conda install pytorch==1.10.1 torchvision torchaudio cudatoolkit=11.3 -c pytorch
conda install pytorch3d -c pytorch3d
conda install -c conda-forge meshplot
conda install -c conda-forge jupyterlab #optional
pip install -r requirements.txt

# kaolin
pip install cython==0.29.20
pip install kaolin==0.12.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-1.10.1_cu113.html
```

