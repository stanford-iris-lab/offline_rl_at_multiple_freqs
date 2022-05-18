#!/bin/bash
#SBATCH --partition=iris-hi
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --exclude=iris4
#SBATCH --job-name="eval"
#SBATCH --time=3-0:0

source /sailhome/kayburns/.bashrc                                                  
# source /sailhome/kayburns/set_cuda_paths.sh                                        
conda deactivate
conda activate py3.7_torch1.8
cd /iris/u/kayburns/continuous-rl/
export PYTHONPATH="$PYTHONPATH:$(pwd)"
cd /iris/u/kayburns/continuous-rl/CQL
export PYTHONPATH="$PYTHONPATH:$(pwd)"
export MJLIB_PATH=/sailhome/kayburns/anaconda3/envs/py3.7_torch1.8/lib/python3.7/site-packages/mujoco_py/binaries/linux/mujoco210/bin/libmujoco210.so

Xvfb :0 &
MUJOCO_GL=egl DISPLAY=:0 python -m SimpleSAC.sac_main \
  --logging.output_dir "./debug/compare_80/03aacbc989474a0892800bb34a9d0bcf/" \
  --logging.online False \
  --env "kitchen-complete-v0" \
  --dt 80 \
  --max_traj_length 500 \
  --seed 42 \
  --device 'cuda' \
  --n_train_step_per_epoch 0 \
  --n_epochs 1 \
  --eval_n_trajs 1 \
  --load_model_from_path "/iris/u/kayburns/continuous-rl/CQL/experiments/collect/kitchen-complete-v0/03aacbc989474a0892800bb34a9d0bcf/model_r3.799999952316284_epoch149.pkl" \
  --save_model False
