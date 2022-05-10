import random
import pprint
import time
import uuid
import tempfile
import os
from copy import copy
from socket import gethostname
import pickle

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvas

import absl.flags
from absl import logging
from ml_collections import ConfigDict
from ml_collections.config_flags import config_flags
from ml_collections.config_dict import config_dict

from moviepy.editor import ImageSequenceClip

import wandb

import torch


class Timer(object):

    def __init__(self):
        self._time = None

    def __enter__(self):
        self._start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self._time = time.time() - self._start_time

    def __call__(self):
        return self._time


class WandBLogger(object):

    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.online = False
        config.prefix = 'SimpleSAC'
        config.project = 'sac'
        config.output_dir = '/tmp/SimpleSAC'
        config.random_delay = 0.0
        config.experiment_id = config_dict.placeholder(str)
        config.anonymous = config_dict.placeholder(str)
        config.notes = config_dict.placeholder(str)

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config
    
    @staticmethod
    def plot(y_values):
        x_values = np.arange(0, len(y_values)/10, .1)
        data = [[x, y] for (x, y) in zip(x_values, y_values)]
        table = wandb.Table(data=data, columns = ["x", "y"])
        return wandb.plot.line(table, "x", "y")

    def __init__(self, config, variant):
        self.config = self.get_default_config(config)

        if self.config.experiment_id is None:
            self.config.experiment_id = uuid.uuid4().hex

        if self.config.prefix != '':
            self.config.project = '{}--{}'.format(self.config.prefix, self.config.project)

        if self.config.output_dir == '':
            self.config.output_dir = tempfile.mkdtemp()
        else:
            self.config.output_dir = os.path.join(self.config.output_dir, self.config.experiment_id)
            os.makedirs(self.config.output_dir, exist_ok=True)

        self._variant = copy(variant)

        if 'hostname' not in self._variant:
            self._variant['hostname'] = gethostname()

        if self.config.random_delay > 0:
            time.sleep(np.random.uniform(0, self.config.random_delay))

        self.run = wandb.init(
            reinit=True,
            config=self._variant,
            project=self.config.project,
            dir=self.config.output_dir,
            id=self.config.experiment_id,
            anonymous=self.config.anonymous,
            notes=self.config.notes,
            settings=wandb.Settings(
                start_method="thread",
                _disable_stats=True,
            ),
            # mode='online'
            mode='online' if self.config.online else 'offline',
        )

    def log(self, *args, **kwargs):
        self.run.log(*args, **kwargs)

    def save_pickle(self, obj, filename):
        with open(os.path.join(self.config.output_dir, filename), 'wb') as fout:
            pickle.dump(obj, fout)
    
    def save_image(self, image, filename):
        plt.imshow(image)
        plt.savefig(os.path.join(self.config.output_dir, filename))
    
    def load_pickle(self, loaddir):
        with open(os.path.join(loaddir, 'model.pkl'), 'rb') as fout:
            return pickle.load(fout)
    
    def load_pickle_from_filename(self, loadpath):
        with open(loadpath, 'rb') as fout:
            return pickle.load(fout)

    @property
    def experiment_id(self):
        return self.config.experiment_id

    @property
    def variant(self):
        return self.config.variant

    @property
    def output_dir(self):
        return self.config.output_dir


def define_flags_with_default(**kwargs):
    for key, val in kwargs.items():
        if isinstance(val, ConfigDict):
            config_flags.DEFINE_config_dict(key, val)
        elif isinstance(val, bool):
            # Note that True and False are instances of int.
            absl.flags.DEFINE_bool(key, val, 'automatically defined flag')
        elif isinstance(val, int):
            absl.flags.DEFINE_integer(key, val, 'automatically defined flag')
        elif isinstance(val, float):
            absl.flags.DEFINE_float(key, val, 'automatically defined flag')
        elif isinstance(val, str):
            absl.flags.DEFINE_string(key, val, 'automatically defined flag')
        else:
            raise ValueError('Incorrect value type')
    return kwargs


def set_random_seed(seed):
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    random.seed(seed)


def print_flags(flags, flags_def):
    logging.info(
        'Running training with hyperparameters: \n{}'.format(
            pprint.pformat(
                ['{}: {}'.format(key, val) for key, val in get_user_flags(flags, flags_def).items()]
            )
        )
    )


def get_user_flags(flags, flags_def):
    output = {}
    for key in flags_def:
        val = getattr(flags, key)
        if isinstance(val, ConfigDict):
            output.update(flatten_config_dict(val, prefix=key))
        else:
            output[key] = val

    return output


def flatten_config_dict(config, prefix=None):
    output = {}
    for key, val in config.items():
        if isinstance(val, ConfigDict):
            output.update(flatten_config_dict(val, prefix=key))
        else:
            if prefix is not None:
                output['{}.{}'.format(prefix, key)] = val
            else:
                output[key] = val
    return output



def prefix_metrics(metrics, prefix):
    return {
        '{}/{}'.format(prefix, key): value for key, value in metrics.items()
    }

# Pendulum visualizations adapted from https://github.com/ctallec/continuous-rl
def th_to_arr(tens: torch.Tensor) -> np.ndarray:
    """Tensorable to numpy array."""
    return tens.cpu().detach().numpy()

def arr_to_th(arr) -> torch.Tensor:
    """Arrayable to tensor."""

    return torch.from_numpy(arr).float().to('cuda')

def generate_pendulum_visualization(policy, qf1, qf2, logger, filename, dt_feat, dt):
    nb_pixels = 50
    theta_space = np.linspace(-np.pi, np.pi, nb_pixels)
    dtheta_space = np.linspace(-10, 10, nb_pixels)
    theta, dtheta = np.meshgrid(theta_space, dtheta_space)
    state_space = np.stack([np.cos(theta), np.sin(theta), dtheta], axis=-1)
    target_shape = state_space.shape[:2]
    state_space = arr_to_th(state_space).reshape(-1, 3)

    observation = state_space
    
    if dt_feat:
        dt_feat = (torch.ones((state_space.shape[0], 1)) * dt).cuda()
        observation = torch.hstack([state_space, dt_feat])
    actions = policy(observation)[0]
    values = qf1(observation, actions).reshape(target_shape).squeeze()

    # normalize values and visualize with plasma colormap
    # values = (values - values.mean()) / values.std()
    # for pendulum consistency
    max_R = 10000
    values = (values - (max_R/2)) / (max_R/2)
    vis_values = th_to_arr(values)
    vis_values = plt.get_cmap("plasma")(vis_values)
    logger.save_image(vis_values, filename)


def vid_from_frames(imgs, output_file):
    imgs = [img.squeeze() for img in np.split(imgs, imgs.shape[0])]
    clip = ImageSequenceClip(imgs, fps=50)
    if output_file.endswith('.gif'):
        clip.write_gif(output_file)
    if output_file.endswith('.mp4'):
        clip.write_videofile(output_file)

def np_unstack(array, axis):
    arr = np.split(array, array.shape[axis], axis)
    arr = [a.squeeze() for a in arr]
    return arr

def plot_q_over_traj(q_estimates, rewards, images, output_file):
    """
    Args:
        - rewards: list of r
        - list of q1(s, a), q2(s,a) from traj
    Returns:
        - 
    """
    q_estimates_np = np.stack(q_estimates, 1)

    fig, axs = plt.subplots(3, 1)
    canvas = FigureCanvas(fig)
    plt.xlim([0, len(q_estimates)])

    # assume image in T, C, H, W shape
    assert len(images.shape) == 4
    assert images.shape[-1] == 3

    interval = images.shape[0] // 4
    sel_images = images[::interval]
    sel_images = np.concatenate(np_unstack(sel_images, 0), 1)

    axs[0].imshow(sel_images)
    axs[1].plot(q_estimates_np[:, 0], linestyle='--', marker='o')
    axs[1].plot(q_estimates_np[:, 1], linestyle='--', marker='o')
    axs[1].set_ylabel('q values')
    axs[2].plot(rewards, linestyle='--', marker='o')
    axs[2].set_ylabel('rewards')
    axs[2].set_xlim([0, len(rewards)])

    plt.tight_layout()

    canvas.draw() 
    out_image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    out_image = out_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.savefig(output_file)
    return out_image

