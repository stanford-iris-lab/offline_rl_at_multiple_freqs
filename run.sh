#!/bin/bash
#SBATCH --partition=iris-hi
#SBATCH --exclude=iris-hp-z8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --job-name="cql"
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
DISPLAY=:0 python -m SimpleSAC.conservative_sac_main \
  --env "pendulum_${4}" \
  --logging.output_dir "./experiments/${4}/" \
  --load_model "/iris/u/kayburns/continuous-rl/CQL/experiments/.02/61750f982c6243c9898c572d835cf79c/" \
  --cql.buffer_file "/iris/u/kayburns/continuous-rl/dau/logdir/continuous_pendulum_sparse1/cdau/half_buffer_1_${4}/data0.h5py" \
  --max_traj_length 1000 \
  --n_train_step_per_epoch 0 \
  --n_epochs 1 \
  --eval_n_trajs 1 \
  --device 'cuda' \
#   --save_model True
python vid_from_npz.py \
  --npz_file ./movie.npz \
  --fps 50 --output_file ./vid.mp4

# 0.001: 5 1e-4 3e-4 .001 .9999
# 0.01: 5 1e-4 3e-4 .01 .999
# 0.1: 1 1e-4 3e-4 .1 .99
