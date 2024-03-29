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
# DISPLAY=:0 python -m SimpleSAC.conservative_sac_main \
#   --env "pendulum_${4}" \
#   --logging.output_dir "./experiments/${4}/" \
#   --logging.online True \
#   --cql.cql_min_q_weight ${1} \
#   --cql.policy_lr ${2} \
#   --cql.qf_lr ${3} \
#   --cql.discount ${5} \
#   --cql.buffer_file "/iris/u/kayburns/continuous-rl/dau/logdir/continuous_pendulum_sparse1/cdau/half_buffer_0_${4}/data0.h5py" \
#   --device 'cuda' \
#   --save_model True

# off-line trainning mujoco
MUJOCO_GL=egl DISPLAY=:0 python -m SimpleSAC.conservative_sac_main \
  --logging.output_dir "./experiments/mujoco/${8}/" \
  --logging.online True \
  --load_model "/iris/u/kayburns/continuous-rl/CQL/experiments/mujoco/kitchen-complete-v0/0303f0c18c494c56aa7dda17218cb3f7/" \
  --env "${8}" \
  --max_traj_length ${6} \
  --cql.policy_lr ${2} \
  --seed ${7} \
  --device 'cuda' \
  --save_model True \
  --cql.cql_min_q_weight ${1} \
  --cql.qf_lr ${3} \
  --cql.discount ${5} \
  --cql.buffer_file "/iris/u/kayburns/continuous-rl/CQL/experiments/collect/door-open-v2-goal-observable/67fa1c8c44a94062b7b6d1a8914d176a/buffer.h5py"


# 0.001: 5 1e-4 3e-4 .001 .9999
# 0.01: 5 1e-4 3e-4 .01 .999
# 0.1: 1 1e-4 3e-4 .1 .99

# half_buffer_1 for .02
