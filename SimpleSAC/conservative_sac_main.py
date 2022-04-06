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

from .conservative_sac import ConservativeSAC
from .replay_buffer import batch_to_torch, subsample_batch_n, subsample_flat_batch_n, load_dataset, load_d4rl_dataset
from .model import TanhGaussianPolicy, FullyConnectedQFunction, SamplerPolicy
from .sampler import TrajSampler
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
    sparse=True,

    reward_scale=1.0,
    reward_bias=0.0,

    policy_arch='256-256',
    qf_arch='256-256',
    orthogonal_init=False,
    policy_log_std_multiplier=1.0,
    policy_log_std_offset=-1.0,

    n_epochs=800,
    n_train_step_per_epoch=1000,
    eval_period=10,
    eval_n_trajs=5,
    load_model='',
    visualize_traj=False,
    # N_steps=.02,
    # N_datapoints=250000,
    dt_feat=True,
    use_pretrained_q_target=False,
    pretrained_target_path='',
    shared_q_target=False,
    max_q_target=False,
    # use_pretrained_q_target=True,
    # pretrained_target_path='/iris/u/kayburns/continuous-rl/CQL/experiments/.02/aec001f95d094fa598456707e8c81814/',

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
        eval_samplers[.005] = TrajSampler(Walker(.005))
        eval_samplers[.02] = TrajSampler(Walker(.02))
        eval_samplers[.01] = TrajSampler(Walker(.01))
        # eval_samplers[.005] = TrajSampler(Walker(.005), int(100 * (1/.005)))
        # eval_samplers[.02] = TrajSampler(Walker(.02), int(100 * (1/.02)))
        # eval_samplers[.01] = TrajSampler(Walker(.01), int(100 * (1/.01)))

        datasets = {}
        for dt in [.01, .02, .005]:
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
        datasets, eval_samplers = {}, {}
        for dt in [.01, .02, .005]:
            env = gym.make('Pendulum-v1').unwrapped
            env.dt = dt
            eval_samplers[dt] = TrajSampler(WrapContinuousPendulumSparse(env), FLAGS.max_traj_length)
            # eval_samplers[dt] = TrajSampler(WrapContinuousPendulumSparse(env), int(100 * (1/dt)))
            dataset = load_dataset(f"/iris/u/kayburns/continuous-rl/dau/logdir/continuous_pendulum_sparse1/cdau/half_buffer_0_{str(dt)[1:]}/data0.h5py")
            dataset['rewards'] = dataset['rewards'] * FLAGS.reward_scale + FLAGS.reward_bias
            datasets[dt] = dataset
    elif "door-open-v2-goal-observable" in FLAGS.env:
        # find correct buffer file
        buffers = {
            1: "/iris/u/kayburns/continuous-rl/CQL/experiments/collect/door-open-v2-goal-observable/f87d142ac7e54d659d999cba3e5e5421/buffer.h5py",
            2: "/iris/u/kayburns/continuous-rl/CQL/experiments/collect/door-open-v2-goal-observable/8690f0c73f7a4b94b1c7dbc3330174eb/buffer.h5py",
            5: "/iris/u/kayburns/continuous-rl/CQL/experiments/collect/door-open-v2-goal-observable/67fa1c8c44a94062b7b6d1a8914d176a/buffer.h5py",
            10: "/iris/u/kayburns/continuous-rl/CQL/experiments/collect/door-open-v2-goal-observable/b6842bc3810641f6868fb42a242fe059/buffer.h5py"
        }
        datasets, eval_samplers = {}, {}

        dts = list(buffers.keys())
        for dt in dts:
            # load environment
            env = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[FLAGS.env](seed=FLAGS.seed)
            env.frame_skip = dt
            assert env.dt == dt * .00125
            eval_samplers[dt] = TrajSampler(env, FLAGS.max_traj_length)

            # fetch dataset
            dataset = load_dataset(buffers[dt])
            if FLAGS.sparse:
                dataset['rewards'] = (dataset['rewards'] == 10.0 * (dt/10)).astype('float32')
            datasets[dt] = dataset
    elif "drawer-open-v2-goal-observable" in FLAGS.env:
        # find correct buffer file
        buffers = {
            1: "/iris/u/kayburns/continuous-rl/CQL/experiments/collect/drawer-open-v2-goal-observable/a3240368c8534dc4bbae373acd166008/buffer.h5py",
            2: "/iris/u/kayburns/continuous-rl/CQL/experiments/collect/drawer-open-v2-goal-observable/fcbc6ee5141749a7a3bd3224e68c5f06/buffer.h5py",
            5: "/iris/u/kayburns/continuous-rl/CQL/experiments/collect/drawer-open-v2-goal-observable/85ffe681bb424d219305ebfed7d30581/buffer.h5py",
            10: "/iris/u/kayburns/continuous-rl/CQL/experiments/collect/drawer-open-v2-goal-observable/180be33816c24878a114d3c9816d65d5/buffer.h5py"
        }
        datasets, eval_samplers = {}, {}

        dts = list(buffers.keys())
        for dt in dts:
            # load environment
            env = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[FLAGS.env](seed=FLAGS.seed)
            env.frame_skip = dt
            assert env.dt == dt * .00125
            eval_samplers[dt] = TrajSampler(env, FLAGS.max_traj_length)

            # fetch dataset
            dataset = load_dataset(buffers[dt])
            if FLAGS.sparse:
                dataset['rewards'] = (dataset['rewards'] == 10.0 * (dt/10)).astype('float32')
            datasets[dt] = dataset
    elif 'kitchen' in FLAGS.env:
        datasets, eval_samplers = {}, {}
        env = gym.make(FLAGS.env)
        datasets[40] = load_d4rl_dataset(env)
        datasets[80] = load_dataset(
            '/iris/u/kayburns/continuous-rl/CQL/experiments/collect/kitchen-complete-v0/6027408585064c61bf5b356b14f96607/buffer.h5py')
        
        env40 = gym.make(FLAGS.env)
        assert env40.dt == 40 * .002
        eval_samplers[40] = TrajSampler(env40.unwrapped, FLAGS.max_traj_length)

        env80 = gym.make(FLAGS.env)
        env80.frame_skip = 80
        env80.dt = 80 * .002
        assert env80.dt == 80 * .002
        eval_samplers[80] = TrajSampler(env80.unwrapped, FLAGS.max_traj_length)
    else:
        eval_sampler = TrajSampler(gym.make(FLAGS.env).unwrapped, FLAGS.max_traj_length) # TODO

    # dataset['rewards'] = dataset['rewards'] * FLAGS.reward_scale + FLAGS.reward_bias

    if FLAGS.load_model:
        loaded_model = wandb_logger.load_pickle(FLAGS.load_model)
        print(f"Loaded model from epoch {loaded_model['epoch']}")
        sac = loaded_model['sac']
        policy = sac.policy
    else:
        if FLAGS.dt_feat:
            obs_shape = list(eval_samplers.values())[0].env.observation_space.shape[0]+1
        else:
            obs_shape = list(eval_samplers.values())[0].env.observation_space.shape[0]
        action_shape = list(eval_samplers.values())[0].env.action_space.shape

        policy = TanhGaussianPolicy(
            obs_shape,
            action_shape[0],
            arch=FLAGS.policy_arch,
            log_std_multiplier=FLAGS.policy_log_std_multiplier,
            log_std_offset=FLAGS.policy_log_std_offset,
            orthogonal_init=FLAGS.orthogonal_init,
        )

        qf1 = FullyConnectedQFunction(
            obs_shape,
            action_shape[0],
            arch=FLAGS.qf_arch,
            orthogonal_init=FLAGS.orthogonal_init,
        )

        qf2 = FullyConnectedQFunction(
            obs_shape,
            action_shape[0],
            arch=FLAGS.qf_arch,
            orthogonal_init=FLAGS.orthogonal_init,
        )

        if FLAGS.use_pretrained_q_target:
            # pretrained_target_path = '/iris/u/kayburns/continuous-rl/CQL/experiments/pendulum/mix_pcond/ad794d1af211434f9cab06cfd0784b5f/'
            loaded_model = wandb_logger.load_pickle(FLAGS.pretrained_target_path)
            print(f"Loaded model from epoch {loaded_model['epoch']}")
            sac = loaded_model['sac']
            target_qf1 = sac.target_qf1
            target_qf2 = sac.target_qf2
        else:
            target_qf1 = deepcopy(qf1)
            target_qf2 = deepcopy(qf2)

        if FLAGS.cql.target_entropy >= 0.0:
            FLAGS.cql.target_entropy = -np.prod(action_shape).item()

        update_target = not FLAGS.use_pretrained_q_target
        sac = ConservativeSAC(FLAGS.cql, policy, qf1, qf2, target_qf1, target_qf2, update_target=update_target)
    sac.torch_to_device(FLAGS.device)

    sampler_policy = SamplerPolicy(policy, FLAGS.device)

    viskit_metrics = {}
    dts = sorted(list(eval_samplers.keys()))
    for epoch in range(FLAGS.n_epochs):
        metrics = {'epoch': epoch}

        with Timer() as train_timer:
            for batch_idx in range(FLAGS.n_train_step_per_epoch):
                per_dataset_batch_size = int(FLAGS.batch_size / len(dts))

                batch_dts = []
                # max_steps = int(FLAGS.N_steps / min(dts))
                max_steps = 1
                for dt in dts:
                    # batch_dt is N, 1, D
                    if dt == 40:
                        batch_dt = subsample_flat_batch_n(
                            datasets[dt], per_dataset_batch_size, max_steps)
                    else:
                        batch_dt = subsample_batch_n(
                            datasets[dt], per_dataset_batch_size, max_steps)
                    if FLAGS.dt_feat:
                        dt_feat = np.ones((per_dataset_batch_size, max_steps, 1))*dt
                        norm_dt = (dt_feat - np.mean(dts)) / np.std(dts)
                        batch_dt['observations'] = np.concatenate([
                            batch_dt['observations'], norm_dt], axis=2
                        ).astype(np.float32)
                        batch_dt['next_observations'] = np.concatenate([
                            batch_dt['next_observations'], norm_dt], axis=2
                        ).astype(np.float32)
                    batch_dts.append(batch_dt)

                # create a batch which samples equally from each buffer
                batch = {}
                for k in batch_dts[0].keys():
                    batch[k] = np.concatenate([b[k] for b in batch_dts], axis=0)
                batch = batch_to_torch(batch, FLAGS.device)
                # n_steps = torch.Tensor([FLAGS.N_steps/dt for dt in dts])
                n_steps = torch.Tensor([1 for dt in dts])
                n_steps = n_steps.repeat_interleave(per_dataset_batch_size)
                # TODO weird: this is replicating the same indexing per_dataset_batch_size times
                if FLAGS.shared_q_target:
                    # batch['next_observations'][:,(n_steps-1).long(),-1] = max(dts)
                    batch['next_observations'][:,(n_steps-1).long(),-1] = (max(dts) - np.mean(dts)) / np.std(dts)
                discount_arr = torch.Tensor([FLAGS.cql.discount ** (dt/max(dts)) for dt in dts]).cuda()
                discount_arr =  discount_arr.repeat_interleave(per_dataset_batch_size)
                metrics.update(prefix_metrics(sac.train(batch, n_steps, discount_arr, FLAGS.max_q_target), 'sac'))

        with Timer() as eval_timer:
            for dt, eval_sampler in eval_samplers.items():
                if epoch == 0 or (epoch + 1) % FLAGS.eval_period == 0:
                    # my_seed = eval_sampler._env.seed(FLAGS.seed)
                    video = epoch == 0 or (epoch + 1) % (FLAGS.eval_period * 10) == 0
                    output_file = os.path.join(
                        wandb_logger.config.output_dir, f'eval_dt_{dt}_{epoch}.gif')

                    norm_dt = (dt - np.mean(dts)) / np.std(dts)
                    trajs = eval_sampler.sample(
                        sampler_policy, FLAGS.eval_n_trajs, FLAGS.dt_feat, norm_dt,
                        deterministic=True, video=video, output_file=output_file
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
                            norm_dt = (dt - np.mean(dts)) / np.std(dts)
                            generate_pendulum_visualization(
                                sac.policy, sac.qf1, sac.qf2, wandb_logger,
                                f'val_dt{dt}_epoch{epoch}.png', FLAGS.dt_feat, norm_dt)

                    if "goal-observable" in FLAGS.env:
                        metrics[f'max_success_{dt}'] = np.mean([np.max(t['successes']) for t in trajs])
                        metrics[f'final_state_success_{dt}'] = np.mean([t['successes'][-1] for t in trajs])
                    metrics[f'average_return_{dt}'] = np.mean([np.sum(t['rewards']) for t in trajs])
                    metrics[f'average_traj_length_{dt}'] = np.mean([len(t['rewards']) for t in trajs])
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
