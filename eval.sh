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
  --env "walker_.01" \
  --logging.output_dir "./debug/" \
  --cql.buffer_file "/iris/u/kayburns/continuous-rl/dau/logdir/bipedal_walker/cdau/medium_buffer_.1/data0.h5py" \
  --n_epochs 1 \
  --n_train_step_per_epoch 0 \
  --device 'cuda' \
  --eval_n_trajs 1 \
  --max_traj_length 10000 \
  --load_model 'experiments/.1/814e9a2fe0f34f2082d2a4cb0e054956/'
python vid_from_npz.py \
  --npz_file ./movie.npz \
  --fps 100 --output_file ./vid.mp4

