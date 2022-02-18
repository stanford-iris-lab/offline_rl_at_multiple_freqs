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
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE

FLAGS_DEF = define_flags_with_default(
    env='halfcheetah-medium-v2',
    max_traj_length=1000,
    seed=42,
    device='cpu',
    save_model=False,
    batch_size=256,
    sparse=False,

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
    visualize_traj=False,

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
        dt = float(FLAGS.env.split('_')[1])
        eval_sampler = TrajSampler(Walker(dt), FLAGS.max_traj_length) # TODO
    elif "pendulum_" in FLAGS.env:
        dt = float(FLAGS.env.split('_')[1])
        env = gym.make('Pendulum-v1').unwrapped
        env.dt = dt
        eval_sampler = TrajSampler(WrapContinuousPendulumSparse(env), FLAGS.max_traj_length)
    elif "goal-observable" in FLAGS.env:
        env_name, fs = FLAGS.env.split('_')
        fs = int(fs)
        env = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[env_name](seed=FLAGS.seed)
        env.frame_skip = fs
        assert env.dt == fs * .00125
        eval_sampler = TrajSampler(env, FLAGS.max_traj_length)
        # find correct buffer file
        buffers = {
            1: "/iris/u/kayburns/continuous-rl/CQL/experiments/collect/door-open-v2-goal-observable/f87d142ac7e54d659d999cba3e5e5421/buffer.h5py",
            2: "/iris/u/kayburns/continuous-rl/CQL/experiments/collect/door-open-v2-goal-observable/8690f0c73f7a4b94b1c7dbc3330174eb/buffer.h5py",
            5: "/iris/u/kayburns/continuous-rl/CQL/experiments/collect/door-open-v2-goal-observable/67fa1c8c44a94062b7b6d1a8914d176a/buffer.h5py",
            10: "/iris/u/kayburns/continuous-rl/CQL/experiments/collect/door-open-v2-goal-observable/b6842bc3810641f6868fb42a242fe059/buffer.h5py"
        }
        FLAGS.cql.buffer_file = buffers[fs]
    else:
        eval_sampler = TrajSampler(gym.make(FLAGS.env).unwrapped, FLAGS.max_traj_length) # TODO

    dataset = load_dataset(FLAGS.cql.buffer_file)
    if "goal-observable" in FLAGS.env:
        for k, v in dataset.items():
            dataset[k] = v[:500000]
        if FLAGS.sparse:
            dataset['rewards'] = (dataset['rewards'] == 10.0 * (fs/10)).astype('float32')
    dataset['rewards'] = dataset['rewards'] * FLAGS.reward_scale + FLAGS.reward_bias

    if FLAGS.load_model:
        loaded_model = wandb_logger.load_pickle(FLAGS.load_model)
        print(f"Loaded model from epoch {loaded_model['epoch']}")
        sac = loaded_model['sac']
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
                # my_seed = eval_sampler._env.seed(FLAGS.seed)
                video = epoch == 0 or (epoch + 1) % (FLAGS.eval_period * 10) == 0
                output_file = os.path.join(wandb_logger.config.output_dir, f'eval_{epoch}.gif')
                trajs = eval_sampler.sample(
                    sampler_policy, FLAGS.eval_n_trajs, deterministic=True, fs=fs,
                    video=video, output_file=output_file
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

                if "goal-observable" in FLAGS.env:
                    metrics['max_success'] = np.mean([np.max(t['successes']) for t in trajs])
                    metrics['final_state_success'] = np.mean([t['successes'][-1] for t in trajs])
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
