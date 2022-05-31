#!/bin/bash
#SBATCH --partition=iris-hi
#SBATCH --exclude=iris-hp-z8,iris1,iris4
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
  --env "pendulum" \
  --logging.output_dir "./ldod/pendulum/" \
  --logging.online True \
  --logging.project 'ldod' \
  --cql.cql_min_q_weight 5 \
  --cql.policy_lr 3e-4 \
  --cql.qf_lr 3e-4 \
  --cql.discount .99 \
  --n_epochs 600 \
  --dt_feat ${1} \
  --seed ${2} \
  --N_steps ${3} \
  --device 'cuda' \
  --save_model True
