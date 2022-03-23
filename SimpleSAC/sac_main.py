import os
import time
from copy import deepcopy
import uuid

import numpy as np
import pprint

import gym
import torch
import d4rl

import absl.app
import absl.flags

from .sac import SAC
from .replay_buffer import ReplayBuffer, batch_to_torch
from .model import TanhGaussianPolicy, FullyConnectedQFunction, SamplerPolicy
from .sampler import StepSampler, TrajSampler
from .utils import Timer, define_flags_with_default, set_random_seed, print_flags, get_user_flags, prefix_metrics
from .utils import WandBLogger
from viskit.logging import logger, setup_logger

from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE


FLAGS_DEF = define_flags_with_default(
    env='door-open-v2-goal-observable', # 'drawer-open-v2-goal-observable',
    max_traj_length=500,
    replay_buffer_size=1000000,
    seed=42,
    device='cpu',
    save_model=False,
    dt=80,

    policy_arch='256-256',
    qf_arch='256-256',
    policy_log_std_multiplier=1.0,
    policy_log_std_offset=-1.0,

    n_epochs=500,
    n_env_steps_per_epoch=1000,
    n_train_step_per_epoch=1000,
    eval_period=10,
    eval_n_trajs=5,

    batch_size=256,

    sac=SAC.get_default_config(),
    logging=WandBLogger.get_default_config(),
)


def main(argv):
    FLAGS = absl.flags.FLAGS

    variant = get_user_flags(FLAGS, FLAGS_DEF)
    wandb_logger = WandBLogger(config=FLAGS.logging, variant=variant)
    setup_logger(
        variant=variant,
        exp_id=wandb_logger.experiment_id,
        seed=FLAGS.seed,
        base_log_dir=FLAGS.logging.output_dir,
        include_exp_prefix_sub_dir=False
    )

    set_random_seed(FLAGS.seed)

    if 'goal-observable' in FLAGS.env:
        train_env = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[FLAGS.env](seed=FLAGS.seed-1)
        train_env.frame_skip = FLAGS.dt
        test_env = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[FLAGS.env](seed=FLAGS.seed)
        test_env.frame_skip = FLAGS.dt
        assert train_env.dt == FLAGS.dt * .00125
        assert test_env.dt == FLAGS.dt * .00125
        train_sampler = StepSampler(train_env.unwrapped, FLAGS.max_traj_length)
        eval_sampler = TrajSampler(test_env.unwrapped, FLAGS.max_traj_length)
    elif 'kitchen' in FLAGS.env:
        train_env = gym.make(FLAGS.env).unwrapped
        train_env.frame_skip = FLAGS.dt
        test_env = gym.make(FLAGS.env).unwrapped
        test_env.frame_skip = FLAGS.dt
        assert train_env.dt == FLAGS.dt * .002
        assert test_env.dt == FLAGS.dt * .002
        train_sampler = StepSampler(train_env, FLAGS.max_traj_length)
        eval_sampler = TrajSampler(test_env, FLAGS.max_traj_length)
    else:
        train_sampler = StepSampler(gym.make(FLAGS.env).unwrapped, FLAGS.max_traj_length)
        eval_sampler = TrajSampler(gym.make(FLAGS.env).unwrapped, FLAGS.max_traj_length)

    replay_buffer = ReplayBuffer(FLAGS.replay_buffer_size)

    policy = TanhGaussianPolicy(
        train_sampler.env.observation_space.shape[0],
        train_sampler.env.action_space.shape[0],
        FLAGS.policy_arch,
        log_std_multiplier=FLAGS.policy_log_std_multiplier,
        log_std_offset=FLAGS.policy_log_std_offset,
    )

    qf1 = FullyConnectedQFunction(
        train_sampler.env.observation_space.shape[0],
        train_sampler.env.action_space.shape[0],
        FLAGS.qf_arch
    )
    target_qf1 = deepcopy(qf1)

    qf2 = FullyConnectedQFunction(
        train_sampler.env.observation_space.shape[0],
        train_sampler.env.action_space.shape[0],
        FLAGS.qf_arch
    )
    target_qf2 = deepcopy(qf2)

    if FLAGS.sac.target_entropy >= 0.0:
        FLAGS.sac.target_entropy = -np.prod(eval_sampler.env.action_space.shape).item()

    sac = SAC(FLAGS.sac, policy, qf1, qf2, target_qf1, target_qf2)
    sac.torch_to_device(FLAGS.device)

    sampler_policy = SamplerPolicy(policy, FLAGS.device)

    viskit_metrics = {}
    for epoch in range(FLAGS.n_epochs):
        metrics = {}
        with Timer() as rollout_timer:
            train_sampler.sample(
                sampler_policy, FLAGS.n_env_steps_per_epoch,
                deterministic=False, replay_buffer=replay_buffer
            )
            metrics['env_steps'] = replay_buffer.total_steps
            metrics['epoch'] = epoch

        with Timer() as train_timer:
            for batch_idx in range(FLAGS.n_train_step_per_epoch):
                batch = batch_to_torch(replay_buffer.sample(FLAGS.batch_size), FLAGS.device)
                if batch_idx + 1 == FLAGS.n_train_step_per_epoch:
                    metrics.update(
                        prefix_metrics(sac.train(batch), 'sac')
                    )
                else:
                    sac.train(batch)

        with Timer() as eval_timer:
            if epoch == 0 or (epoch + 1) % FLAGS.eval_period == 0:
                video = epoch == 0 or (epoch + 1) % (FLAGS.eval_period * 10) == 0
                output_file = os.path.join(wandb_logger.config.output_dir, f'eval_{epoch}.gif')
                trajs = eval_sampler.sample(
                    sampler_policy, FLAGS.eval_n_trajs, False, deterministic=True,
                    video=video, output_file=output_file
                )

                metrics['average_return'] = np.mean([np.sum(t['rewards']) for t in trajs])
                metrics['average_traj_length'] = np.mean([len(t['rewards']) for t in trajs])
                if 'goal-observable' in FLAGS.env:
                    metrics['max_success'] = np.mean([np.max(t['successes']) for t in trajs])
                    metrics['final_state_success'] = np.mean([t['successes'][-1] for t in trajs])

                if FLAGS.save_model:
                    save_data = {'sac': sac, 'variant': variant, 'epoch': epoch}
                    wandb_logger.save_pickle(save_data, 'model.pkl')

        metrics['rollout_time'] = rollout_timer()
        metrics['train_time'] = train_timer()
        metrics['eval_time'] = eval_timer()
        metrics['epoch_time'] = rollout_timer() + train_timer() + eval_timer()
        wandb_logger.log(metrics)
        viskit_metrics.update(metrics)
        logger.record_dict(metrics)
        logger.dump_tabular(with_prefix=False, with_timestamp=False)

    if FLAGS.save_model:
        save_data = {'sac': sac, 'variant': variant, 'epoch': epoch}
        wandb_logger.save_pickle(save_data, 'model.pkl')
        replay_buffer.store(os.path.join(wandb_logger.config.output_dir, 'buffer.h5py'))


if __name__ == '__main__':
    absl.app.run(main)
