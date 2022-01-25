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
from .utils import *
from viskit.logging import logger, setup_logger
from dau.code.envs.biped import Walker
from dau.code.envs.wrappers import WrapContinuousPendulumSparse


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

    n_epochs=1200,
    n_train_step_per_epoch=1000,
    eval_period=10,
    eval_n_trajs=5,
    load_model='',
    visualize_traj=False,
    N=.08,

    cql=ConservativeSAC.get_default_config(),
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

    if "walker_" in FLAGS.env:
        eval_samplers = {}
        eval_samplers[.04] = TrajSampler(Walker(.04))
        eval_samplers[.02] = TrajSampler(Walker(.02))
        eval_samplers[.01] = TrajSampler(Walker(.01))
        # eval_samplers[.04] = TrajSampler(Walker(.04), int(100 * (1/.04)))
        # eval_samplers[.02] = TrajSampler(Walker(.02), int(100 * (1/.02)))
        # eval_samplers[.01] = TrajSampler(Walker(.01), int(100 * (1/.01)))

        datasets = {}
        for dt in [.01, .02, .04]:
            if dt == .04:
                dataset = load_dataset(f"/iris/u/kayburns/continuous-rl/dau/logdir/bipedal_walker/cdau/half_buffer_{str(dt)[1:]}/data0.h5py")
            else:
                dataset = load_dataset(f"/iris/u/kayburns/continuous-rl/dau/logdir/bipedal_walker/cdau/half_buffer_0_{str(dt)[1:]}/data0.h5py")
            dataset['rewards'] = dataset['rewards'] * FLAGS.reward_scale + FLAGS.reward_bias
            datasets[dt] = dataset

        # dts = [float(x) for x in FLAGS.env.split('_')[1:]]
        # eval_samplers = {}
        # datasets = {}
        # for dt in dts:
        #     max_traj_length = (1 / dt) * FLAGS.max_traj_length
        #     eval_samplers[dt] = TrajSampler(Walker(dt), max_traj_length)
        #     dataset = load_dataset(FLAGS.cql.buffer_file.format(dt))
        #     dataset['rewards'] = dataset['rewards'] * FLAGS.reward_scale + FLAGS.reward_bias
        #     datasets[dt] = dataset
    elif "pendulum_" in FLAGS.env:
        dt = float(FLAGS.env.split('_')[1])
        env = gym.make('Pendulum-v1').unwrapped
        env.dt = dt

        datasets, eval_samplers = {}, {}
        for dt in [.01, .02, .04]:
            env = gym.make('Pendulum-v1').unwrapped
            env.dt = dt
            eval_samplers[dt] = TrajSampler(WrapContinuousPendulumSparse(env), FLAGS.max_traj_length)
            # eval_samplers[dt] = TrajSampler(WrapContinuousPendulumSparse(env), int(100 * (1/dt)))
            if dt == .02:
                dataset = load_dataset(f"/iris/u/kayburns/continuous-rl/dau/logdir/continuous_pendulum_sparse1/cdau/half_buffer_1_{str(dt)[1:]}/data0.h5py")
            else:
                dataset = load_dataset(f"/iris/u/kayburns/continuous-rl/dau/logdir/continuous_pendulum_sparse1/cdau/half_buffer_0_{str(dt)[1:]}/data0.h5py")
            dataset['rewards'] = dataset['rewards'] * FLAGS.reward_scale + FLAGS.reward_bias
            datasets[dt] = dataset
    else:
        eval_sampler = TrajSampler(gym.make(FLAGS.env).unwrapped, FLAGS.max_traj_length) # TODO
    #dataset = load_dataset(FLAGS.cql.buffer_file) # TODO
    #dataset['rewards'] = dataset['rewards'] * FLAGS.reward_scale + FLAGS.reward_bias

    if FLAGS.load_model:
        loaded_model = wandb_logger.load_pickle(FLAGS.load_model)
        print(f"Loaded model from epoch {loaded_model['epoch']}")
        sac = loaded_model['sac']
        policy = sac.policy
    else:
        policy = TanhGaussianPolicy(
            eval_samplers[.01].env.observation_space.shape[0]+1,
            eval_samplers[.01].env.action_space.shape[0],
            arch=FLAGS.policy_arch,
            log_std_multiplier=FLAGS.policy_log_std_multiplier,
            log_std_offset=FLAGS.policy_log_std_offset,
            orthogonal_init=FLAGS.orthogonal_init,
        )

        qf1 = FullyConnectedQFunction(
            eval_samplers[.01].env.observation_space.shape[0]+1,
            eval_samplers[.01].env.action_space.shape[0],
            arch=FLAGS.qf_arch,
            orthogonal_init=FLAGS.orthogonal_init,
        )
        target_qf1 = deepcopy(qf1)

        qf2 = FullyConnectedQFunction(
            eval_samplers[.01].env.observation_space.shape[0]+1,
            eval_samplers[.01].env.action_space.shape[0],
            arch=FLAGS.qf_arch,
            orthogonal_init=FLAGS.orthogonal_init,
        )
        target_qf2 = deepcopy(qf2)

        if FLAGS.cql.target_entropy >= 0.0:
            FLAGS.cql.target_entropy = -np.prod(eval_samplers[.01].env.action_space.shape).item()

        sac = ConservativeSAC(FLAGS.cql, policy, qf1, qf2, target_qf1, target_qf2)
    sac.torch_to_device(FLAGS.device)

    sampler_policy = SamplerPolicy(policy, FLAGS.device)

    n = FLAGS.N/env.dt
    viskit_metrics = {}
    for epoch in range(FLAGS.n_epochs):
        metrics = {'epoch': epoch}

        with Timer() as train_timer:
            for batch_idx in range(FLAGS.n_train_step_per_epoch):
                per_dataset_batch_size = int(FLAGS.batch_size / 3)

                batch_dts = []
                for dt in [.01, .02, .04]:
                    batch_dt = subsample_batch(datasets[dt], per_dataset_batch_size)
                    # add a feature for dt
                    batch_dt['observations'] = np.hstack([
                        batch_dt['observations'],
                        np.ones((per_dataset_batch_size, 1))*dt]).astype(np.float32)
                    batch_dt['next_observations'] = np.hstack([
                        batch_dt['next_observations'],
                        np.ones((per_dataset_batch_size, 1))*dt]).astype(np.float32)
                    batch_dts.append(batch_dt)

                # create a batch which samples equally from each buffer
                batch = {}
                for k in batch_dts[0].keys():
                    batch[k] = np.concatenate([b[k] for b in batch_dts], axis=0)
                batch = batch_to_torch(batch, FLAGS.device)
                metrics.update(prefix_metrics(sac.train(batch, n), 'sac'))

        with Timer() as eval_timer:
            for dt, eval_sampler in eval_samplers.items():
                if epoch == 0 or (epoch + 1) % FLAGS.eval_period == 0:
                    # my_seed = eval_sampler._env.seed(FLAGS.seed)
                    trajs = eval_sampler.sample(
                        sampler_policy, FLAGS.eval_n_trajs, deterministic=True, video=FLAGS.visualize_traj
                    )

                    if FLAGS.visualize_traj or epoch % 100 == 99:
                        if "walker_" in FLAGS.env:
                            min_traj_len = min([len(t['actions']) for t in trajs])
                            actions = [t['actions'][:min_traj_len] for t in trajs]
                            mean_actions = np.mean(actions, axis=0)
                            metrics['hip0'] = wandb_logger.plot(mean_actions[:,0])
                            metrics['knee0'] = wandb_logger.plot(mean_actions[:,1])
                            metrics['hip1'] = wandb_logger.plot(mean_actions[:,2])
                            metrics['knee1'] = wandb_logger.plot(mean_actions[:,3])
                        elif "pendulum_" in FLAGS.env:
                            generate_pendulum_visualization(
                                sac.policy, sac.qf1, sac.qf2, wandb_logger,
                                f'val_dt{dt}_epoch{epoch}.png', env.dt)

                    metrics[f'average_return_{dt}'] = np.mean([np.sum(t['rewards']) for t in trajs])
                    metrics[f'average_traj_length_{dt}'] = np.mean([len(t['rewards']) for t in trajs])
                    # metrics['average_normalizd_return'] = np.mean(
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
