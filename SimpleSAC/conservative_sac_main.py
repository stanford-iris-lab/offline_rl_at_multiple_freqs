import os
import time
from copy import deepcopy
import uuid

import numpy as np
import pprint

import gym
import torch
# import d4rl

import absl.app
import absl.flags

from .conservative_sac import ConservativeSAC
from .replay_buffer import batch_to_torch, subsample_batch, load_dataset
from .model import TanhGaussianPolicy, FullyConnectedQFunction, SamplerPolicy
from .sampler import StepSampler, TrajSampler
from .utils import Timer, define_flags_with_default, set_random_seed, print_flags, get_user_flags, prefix_metrics
from .utils import WandBLogger
from viskit.logging import logger, setup_logger
from dau.code.envs.biped import Walker


FLAGS_DEF = define_flags_with_default(
    env='halfcheetah-medium-v2',
    max_traj_length=1000,
    seed=42,
    device='cpu',
    save_model=False,
    batch_size=256,

    reward_scale=1.0,
    reward_bias=0.0,

    policy_arch='256-256',
    qf_arch='256-256',
    orthogonal_init=False,
    policy_log_std_multiplier=1.0,
    policy_log_std_offset=-1.0,

    n_epochs=400,
    n_train_step_per_epoch=1000,
    eval_period=10,
    eval_n_trajs=5,
    load_model='',

    cql=ConservativeSAC.get_default_config(),
    logging=WandBLogger.get_default_config(),
)


def main(argv):
    FLAGS = absl.flags.FLAGS
    visualize_traj=True

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

    if "walker_" in FLAGS.env:
        dt = float(FLAGS.env.split('_')[1])
        eval_sampler = TrajSampler(Walker(dt), FLAGS.max_traj_length) # TODO
    else:
        eval_sampler = TrajSampler(gym.make(FLAGS.env).unwrapped, FLAGS.max_traj_length) # TODO
    dataset = load_dataset(FLAGS.cql.buffer_file) # TODO
    dataset['rewards'] = dataset['rewards'] * FLAGS.reward_scale + FLAGS.reward_bias

    if FLAGS.load_model:
        sac = wandb_logger.load_pickle(FLAGS.load_model)['sac']
        policy = sac.policy
    else:
        policy = TanhGaussianPolicy(
            eval_sampler.env.observation_space.shape[0],
            eval_sampler.env.action_space.shape[0],
            arch=FLAGS.policy_arch,
            log_std_multiplier=FLAGS.policy_log_std_multiplier,
            log_std_offset=FLAGS.policy_log_std_offset,
            orthogonal_init=FLAGS.orthogonal_init,
        )

        qf1 = FullyConnectedQFunction(
            eval_sampler.env.observation_space.shape[0],
            eval_sampler.env.action_space.shape[0],
            arch=FLAGS.qf_arch,
            orthogonal_init=FLAGS.orthogonal_init,
        )
        target_qf1 = deepcopy(qf1)

        qf2 = FullyConnectedQFunction(
            eval_sampler.env.observation_space.shape[0],
            eval_sampler.env.action_space.shape[0],
            arch=FLAGS.qf_arch,
            orthogonal_init=FLAGS.orthogonal_init,
        )
        target_qf2 = deepcopy(qf2)

        if FLAGS.cql.target_entropy >= 0.0:
            FLAGS.cql.target_entropy = -np.prod(eval_sampler.env.action_space.shape).item()

        sac = ConservativeSAC(FLAGS.cql, policy, qf1, qf2, target_qf1, target_qf2)
    sac.torch_to_device(FLAGS.device)

    sampler_policy = SamplerPolicy(policy, FLAGS.device)

    viskit_metrics = {}
    for epoch in range(FLAGS.n_epochs):
        metrics = {'epoch': epoch}

        with Timer() as train_timer:
            for batch_idx in range(FLAGS.n_train_step_per_epoch):
                batch = subsample_batch(dataset, FLAGS.batch_size)
                batch = batch_to_torch(batch, FLAGS.device)
                metrics.update(prefix_metrics(sac.train(batch), 'sac'))

        with Timer() as eval_timer:
            if epoch == 0 or (epoch + 1) % FLAGS.eval_period == 0:
                my_seed = eval_sampler._env.seed(FLAGS.seed)
                trajs = eval_sampler.sample(
                    sampler_policy, FLAGS.eval_n_trajs, deterministic=True, video=visualize_traj
                )

                if visualize_traj:
                    min_traj_len = min([len(t['actions']) for t in trajs])
                    actions = [t['actions'][:min_traj_len] for t in trajs]
                    mean_actions = np.mean(actions, axis=0)

                    metrics['hip0'] = wandb_logger.plot(mean_actions[:,0])
                    metrics['knee0'] = wandb_logger.plot(mean_actions[:,1])
                    metrics['hip1'] = wandb_logger.plot(mean_actions[:,2])
                    metrics['knee1'] = wandb_logger.plot(mean_actions[:,3])

                metrics['average_return'] = np.mean([np.sum(t['rewards']) for t in trajs])
                metrics['average_traj_length'] = np.mean([len(t['rewards']) for t in trajs])
                # metrics['average_normalized_return'] = np.mean(
                #     [eval_sampler.env.get_normalized_score(np.sum(t['rewards'])) for t in trajs]
                # ) # TODO
                if FLAGS.save_model:
                    save_data = {'sac': sac, 'variant': variant, 'epoch': epoch}
                    wandb_logger.save_pickle(save_data, 'model.pkl')

        metrics['train_time'] = train_timer()
        metrics['eval_time'] = eval_timer()
        metrics['epoch_time'] = train_timer() + eval_timer()
        wandb_logger.log(metrics)
        viskit_metrics.update(metrics)
        logger.record_dict(viskit_metrics)
        logger.dump_tabular(with_prefix=False, with_timestamp=False)

    if FLAGS.save_model:
        save_data = {'sac': sac, 'variant': variant, 'epoch': epoch}
        wandb_logger.save_pickle(save_data, 'model.pkl')

if __name__ == '__main__':
    absl.app.run(main)
