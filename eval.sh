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
  --cql.buffer_file "/iris/u/kayburns/continuous-rl/dau/logdir/bipedal_walker/cdau/medium_buffer_.1/data0.h5py" \
  --n_epochs 1 \
  --n_train_step_per_epoch 0 \
  --device 'cuda' \
  --eval_n_trajs 1 \
  --max_traj_length 1000 \
  --load_model 'experiments/.04/ff575473fc2541f9bfec0779b74e9486/'
# load options
  # --load_model 'experiments/pendulum/mix_pcond/ad794d1af211434f9cab06cfd0784b5f/' # mixed
  # --load_model 'experiments/.02/aec001f95d094fa598456707e8c81814' # 02 alone
  # --load_model 'experiments/.01/e9a0361454fb4aae989dcc90b8efd14e' # 01 alone

