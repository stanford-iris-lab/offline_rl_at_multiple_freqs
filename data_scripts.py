"""Video from npz"""
import argparse
from moviepy.editor import ImageSequenceClip
import numpy as np

from SimpleSAC.replay_buffer import load_dataset

parser = argparse.ArgumentParser()
parser.add_argument(
    '--buffer_file',
    type=str,
    default="/iris/u/kayburns/continuous-rl/CQL/experiments/collect/door-open-v2-goal-observable/b6842bc3810641f6868fb42a242fe059/buffer.h5py")
args = parser.parse_args()

import pdb; pdb.set_trace()


###### compute success rate
dataset = load_dataset(args.buffer_file)
for k, v in dataset.items():
    dataset[k] = v[:500000]
dones = dataset['dones'].astype(int)
success = dataset['rewards'] == 5.0
sum_success = np.cumsum(success)
sum_after_episode = sum_success[dones]
num_successful_runs = sum_after_episode[1:] - sum_after_episode[:-1] + 1
print(f"{num_successful_runs} / {success.shape[0]} runs successful: {num_successful_runs/success.shape[0]}")
sum_success[dones] - sum_success[:-1]



###### compute avg reward
dataset = load_dataset(args.buffer_file)
rewards = dataset['rewards']
dones = dataset['dones']

episode_end_idxs = dones.nonzero()[0]
num_episodes = episode_end_idxs.size

avg_reward = 0
prev_idx = 0
for idx in episode_end_idxs:
    idx += 1
    avg_reward += rewards[prev_idx:idx].sum() / num_episodes
    prev_idx = idx
# print(avg_reward) 

###### view video
# imgs = np.load(args.npz_file)['arr_0']
# imgs = [img.squeeze() for img in np.split(imgs, imgs.shape[0])]
# clip = ImageSequenceClip(imgs, fps=args.fps)
# if args.output_file.endswith('.gif'):
#     clip.write_gif(args.output_file)
# if args.output_file.endswith('.mp4'):
#     clip.write_videofile(args.output_file)
