#!/bin/bash
#SBATCH --partition=iris-hi
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --nodelist=iris5
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
# DISPLAY=:0 python -m SimpleSAC.conservative_sac_main \
#   --env "walker_.01" \
#   --logging.output_dir "./debug/" \
#   --cql.buffer_file "/iris/u/kayburns/continuous-rl/dau/logdir/bipedal_walker/cdau/medium_buffer_.01/data0.h5py" \
#   --n_epochs 1 \
#   --n_train_step_per_epoch 10 \
#   --device 'cuda' \
#   --eval_n_trajs 1 \
#   --max_traj_length 10000 \
#   --load_model 'experiments/.01/3fa4192a63854b7988dde9e8fa28023b/'
DISPLAY=:0 python -m SimpleSAC.conservative_sac_main \
  --env "pendulum_.02" \
  --logging.output_dir "./debug/" \
  --logging.online False \
  --cql.buffer_file "/iris/u/kayburns/continuous-rl/dau/logdir/continuous_pendulum_sparse1/cdau/half_buffer_0_.04/data0.h5py" \
  --n_epochs 1 \
  --n_train_step_per_epoch 1 \
  --device 'cuda' \
  --eval_n_trajs 1 \
  --max_traj_length 1000 \
  --load_model 'experiments/.01/637f04a5c90a4fd38cb54658ef4dd11f/'
# load options
  # --load_model 'experiments/pendulum/mix_pcond/ad794d1af211434f9cab06cfd0784b5f/' # mixed
  # --load_model 'experiments/.02/aec001f95d094fa598456707e8c81814' # 02 alone
  # --load_model 'experiments/.01/e9a0361454fb4aae989dcc90b8efd14e' # 01 alone
MUJOCO_GL=egl DISPLAY=:0 python -m SimpleSAC.conservative_sac_main \
  --logging.output_dir "./debug/" \
  --logging.online False \
  --env "${8}" \
  --max_traj_length ${6} \
  --cql.policy_lr ${2} \
  --seed ${7} \
  --device 'cuda' \
  --save_model True \
  --cql.cql_min_q_weight ${1} \
  --cql.qf_lr ${3} \
  --cql.discount ${5} \
  --load_model \
  --cql.buffer_file "/iris/u/kayburns/continuous-rl/CQL/experiments/collect/door-open-v2-goal-observable/67fa1c8c44a94062b7b6d1a8914d176a/buffer.h5py"

