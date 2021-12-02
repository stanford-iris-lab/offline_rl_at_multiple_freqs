#!/bin/bash
#SBATCH --partition=iris
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
python -m SimpleSAC.conservative_sac_main \
  --env "walker_${4}" \
  --logging.output_dir "./debug/" \
  --cql.cql_min_q_weight ${1} \
  --cql.policy_lr ${2} \
  --cql.qf_lr ${3} \
  --cql.buffer_file ".1_and_.01" \
  --device='cuda'

  # --logging.output_dir "./experiments/${4}/" \
  #--cql.qf_lr ${3} \
  # --cql.policy_lr ${2} \
  # --env 'BipedalWalker-v3' \
