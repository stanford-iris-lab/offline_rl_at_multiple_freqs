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

# from .sac import SAC
# from .conservative_sac import ConservativeSAC
from .mix_sac import MixSAC
from .replay_buffer import ReplayBuffer, batch_to_torch, load_d4rl_dataset
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
    init_buffer=True,
    seed=42,
    device='cpu',
    save_model=False,
    dt=40,

    policy_arch='256-256',
    qf_arch='256-256',
    policy_log_std_multiplier=1.0,
    policy_log_std_offset=-1.0,

    n_epochs=500,
    n_env_steps_per_epoch=1000,
    n_train_step_per_epoch=1000,
    eval_period=10,
    eval_n_trajs=5,
    load_model_from_path='',
    N_steps=80,

    batch_size=256,

    sac=MixSAC.get_default_config(),
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
        train_env = gym.make('kitchen-complete-v0').unwrapped
        train_env.frame_skip = FLAGS.dt
        test_env = gym.make('kitchen-complete-v0').unwrapped
        test_env.frame_skip = FLAGS.dt
        assert train_env.dt == FLAGS.dt * .002
        assert test_env.dt == FLAGS.dt * .002
        train_sampler = StepSampler(train_env, FLAGS.max_traj_length)
        eval_sampler = TrajSampler(test_env, FLAGS.max_traj_length)
    else:
        train_sampler = StepSampler(gym.make(FLAGS.env).unwrapped, FLAGS.max_traj_length)
        eval_sampler = TrajSampler(gym.make(FLAGS.env).unwrapped, FLAGS.max_traj_length)
    
    replay_buffer = ReplayBuffer(FLAGS.replay_buffer_size)

    data = load_d4rl_dataset(train_env)
    expert_buffer = ReplayBuffer(FLAGS.replay_buffer_size, data=data)
    replay_buffer = ReplayBuffer(FLAGS.replay_buffer_size)

    if FLAGS.sac.target_entropy >= 0.0:
        FLAGS.sac.target_entropy = -np.prod(eval_sampler.env.action_space.shape).item()

    if FLAGS.load_model_from_path:
        loaded_model = wandb_logger.load_pickle_from_filename(
            FLAGS.load_model_from_path)
        print(f"Loaded model from epoch {loaded_model['epoch']}")
        cql = loaded_model['sac']

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

        mix_sac = MixSAC(FLAGS.sac, cql.policy, qf1, qf2, target_qf1, target_qf2)
        # mix_sac = MixSAC(FLAGS.sac, cql.policy, cql.qf1, cql.qf2, cql.target_qf1, cql.target_qf2)
        policy = mix_sac.policy
    else:
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

        # sac = SAC(FLAGS.sac, policy, qf1, qf2, target_qf1, target_qf2)
        # cql = ConservativeSAC(FLAGS.sac, sac.policy, sac.qf1, sac.qf2, sac.target_qf1, sac.target_qf2)

    mix_sac.torch_to_device(FLAGS.device)

    sampler_policy = SamplerPolicy(mix_sac.policy, FLAGS.device)

    viskit_metrics = {}
    dts = [80, 40] # we load the replay buffer in first
    per_dataset_batch_size = FLAGS.batch_size // len(dts)
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
                max_steps = int(FLAGS.N_steps / min(dts))
                # max_steps = 1
                batch = batch_to_torch(replay_buffer.sample_n(FLAGS.batch_size//2, max_steps), FLAGS.device)
                expert_batch = batch_to_torch(expert_buffer.sample_n(FLAGS.batch_size//2, max_steps), FLAGS.device)
                # reshape expert_batch
                # for k, v in expert_batch.items():
                #     v = torch.unsqueeze(v, axis=1)
                #     if len(v.shape) < 3:
                #         v = torch.unsqueeze(v, axis=1)
                #     expert_batch[k] = v
                # reshape batch
                # for k, v in batch.items():
                #     v = torch.unsqueeze(v, axis=1)
                #     if len(v.shape) < 3:
                #         v = torch.unsqueeze(v, axis=1)
                #     batch[k] = v
                # concatenate batches
                mix_batch = {}
                for k in batch.keys():
                    mix_batch[k] = torch.cat([batch[k], expert_batch[k]], axis=0)
                n_steps = torch.Tensor([FLAGS.N_steps/dt for dt in dts])
                # n_steps = torch.Tensor([1 for dt in dts])
                n_steps = n_steps.repeat_interleave(per_dataset_batch_size)
                demo_mask = torch.zeros(FLAGS.batch_size).cuda()
                demo_mask[FLAGS.batch_size//2:] = 1
                discount_arr = torch.Tensor([FLAGS.sac.discount ** (dt/max(dts)) for dt in dts]).cuda()
                discount_arr =  discount_arr.repeat_interleave(per_dataset_batch_size)
                if batch_idx + 1 == FLAGS.n_train_step_per_epoch:
                    metrics.update(
                        prefix_metrics(mix_sac.train(mix_batch, demo_mask, n_steps, discount_arr, False), 'mix_sac')
                    )
                else:
                    mix_sac.train(mix_batch, demo_mask, n_steps, discount_arr, False)

        with Timer() as eval_timer:
            if epoch == 0 or (epoch + 1) % FLAGS.eval_period == 0:
                video = epoch == 0 or (epoch + 1) % (FLAGS.eval_period * 10) == 0
                output_file = os.path.join(wandb_logger.config.output_dir, f'eval_{epoch}.gif')
                trajs = eval_sampler.sample(
                    sampler_policy, FLAGS.eval_n_trajs, False, 0, deterministic=True,
                    video=video, output_file=output_file, qs=[mix_sac.qf1, mix_sac.qf2]
                )

                metrics['average_return'] = np.mean([np.sum(t['rewards']) for t in trajs])
                metrics['average_traj_length'] = np.mean([len(t['rewards']) for t in trajs])
                if 'goal-observable' in FLAGS.env:
                    metrics['max_success'] = np.mean([np.max(t['successes']) for t in trajs])
                    metrics['final_state_success'] = np.mean([t['successes'][-1] for t in trajs])

                if FLAGS.save_model:
                    # if metrics['average_return'] >= 1:
                    #     file_name = f"model_r{metrics['average_return']}_epoch{epoch}"
                    # else:
                    #     file_name = 'model.pkl'
                    file_name = 'model.pkl'
                    save_data = {'sac': mix_sac, 'variant': variant, 'epoch': epoch}
                    wandb_logger.save_pickle(save_data, file_name)

        metrics['rollout_time'] = rollout_timer()
        metrics['train_time'] = train_timer()
        metrics['eval_time'] = eval_timer()
        metrics['epoch_time'] = rollout_timer() + train_timer() + eval_timer()
        wandb_logger.log(metrics)
        viskit_metrics.update(metrics)
        logger.record_dict(metrics)
        logger.dump_tabular(with_prefix=False, with_timestamp=False)

    if FLAGS.save_model:
        save_data = {'sac': mix_sac, 'variant': variant, 'epoch': epoch}
        wandb_logger.save_pickle(save_data, 'model.pkl')
        replay_buffer.store(os.path.join(wandb_logger.config.output_dir, 'buffer.h5py'))


if __name__ == '__main__':
    absl.app.run(main)
