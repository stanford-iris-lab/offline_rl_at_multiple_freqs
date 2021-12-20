"""Video from npz"""
import argparse
from moviepy.editor import ImageSequenceClip
import numpy as np

from SimpleSAC.replay_buffer import load_dataset

parser = argparse.ArgumentParser()
parser.add_argument(
    '--buffer_file',
    type=str,
    default='/iris/u/kayburns/continuous-rl/dau/logdir/bipedal_walker/cdau/half_buffer_0_.02/data0.h5py')
args = parser.parse_args()

import pdb; pdb.set_trace()

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
print(avg_reward) 

# imgs = np.load(args.npz_file)['arr_0']
# imgs = [img.squeeze() for img in np.split(imgs, imgs.shape[0])]
# clip = ImageSequenceClip(imgs, fps=args.fps)
# if args.output_file.endswith('.gif'):
#     clip.write_gif(args.output_file)
# if args.output_file.endswith('.mp4'):
#     clip.write_videofile(args.output_file)
