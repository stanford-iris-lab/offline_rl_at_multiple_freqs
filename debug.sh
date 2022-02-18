#!/bin/bash

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
# MUJOCO_GL=egl DISPLAY=:0 python -m SimpleSAC.sac_main \
#   --logging.output_dir "./debug/" \
#   --n_epochs 1 \
#   --dt 10 \
#   --env "drawer-open-v2-goal-observable" \
#   --n_train_step_per_epoch 1 \
#   --max_traj_length 1000
#   --device 'cuda' \
#   --save_model True 
#   # --load_model 'debug/b77da20b7bf844ce9cfb9893a02e754e/'
MUJOCO_GL=egl DISPLAY=:0 python -m SimpleSAC.conservative_sac_main \
  --logging.output_dir "./debug/" \
  --env "door-open-v2-goal-observable_5" \
  --n_epochs 1 \
  --n_train_step_per_epoch 1 \
  --max_traj_length 10 \
  --cql.policy_lr 1e-3 \
  --seed 4 \
  --device 'cuda' \
  --save_model False \
  --cql.cql_min_q_weight 5.0 \
  --cql.qf_lr 3e-4 \
  --cql.discount .99 \
  --cql.buffer_file "/iris/u/kayburns/continuous-rl/CQL/experiments/collect/door-open-v2-goal-observable/67fa1c8c44a94062b7b6d1a8914d176a/buffer.h5py"
# python -m SimpleSAC.conservative_sac_main \
#   --env "walker_.01" \
#   --logging.output_dir "./debug/" \
#   --cql.buffer_file "/iris/u/kayburns/continuous-rl/dau/logdir/bipedal_walker/cdau/medium_buffer_.01/data0.h5py" \
#   --n_epochs 1 \
#   --n_train_step_per_epoch 10 \
#   --device 'cuda' 
#   # --load_model 'debug/b77da20b7bf844ce9cfb9893a02e754e/'


