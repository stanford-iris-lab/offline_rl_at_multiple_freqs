import os
import numpy as np
import torch

from .utils import vid_from_frames, plot_q_over_traj

class StepSampler(object):

    def __init__(self, env, max_traj_length=1000):
        self.max_traj_length = max_traj_length
        self._env = env
        self._traj_steps = 0
        self._current_observation = self.env.reset()

    def sample(self, policy, n_steps, deterministic=False, replay_buffer=None):
        observations = []
        actions = []
        rewards = []
        next_observations = []
        dones = []

        for _ in range(n_steps):
            self._traj_steps += 1
            observation = self._current_observation
            action = policy(
                np.expand_dims(observation, 0), deterministic=deterministic
            # )[0, :]
            )[0, :] / 2
            next_observation, reward, done, _ = self.env.step(action)
            # reward = reward * (fs/10)
            observations.append(observation)
            # actions.append(action)
            actions.append(action*2)
            rewards.append(reward)
            dones.append(done)
            next_observations.append(next_observation)

            if replay_buffer is not None:
                replay_buffer.add_sample(
                    # observation, action, reward, next_observation, done
                    observation, action*2, reward, next_observation, done
                )

            self._current_observation = next_observation

            if done or self._traj_steps >= self.max_traj_length:
                self._traj_steps = 0
                self._current_observation = self.env.reset()

        return dict(
            observations=np.array(observations, dtype=np.float32),
            actions=np.array(actions, dtype=np.float32),
            rewards=np.array(rewards, dtype=np.float32),
            next_observations=np.array(next_observations, dtype=np.float32),
            dones=np.array(dones, dtype=np.float32),
        )

    @property
    def env(self):
        return self._env


class TrajSampler(object):

    def __init__(self, env, max_traj_length=1000):
        self.max_traj_length = max_traj_length
        self._env = env

    def sample(self, policy, n_trajs, dt_feat, dt, deterministic=False, replay_buffer=None, video=False, output_file='', qs=None):
        trajs = []
        for traj in range(n_trajs):
            observations = []
            actions = []
            rewards = []
            next_observations = []
            dones = []
            successes = []
            if video and traj == 0:
                imgs = []

            observation = self.env.reset()

            # if you want to play back actions at different dt, uncomment
            # import pickle
            # old_actions = pickle.load(open('actions.pkl', 'rb'))
            for _ in range(self.max_traj_length):
                action = policy(
                    np.expand_dims(observation, 0), deterministic=deterministic
                )[0, :]
                action = action/2
                # action = action
                # if you want to test action repeats, uncomment
                # if _ % 10 == 0:
                #     action = policy(
                #         np.expand_dims(observation, 0), deterministic=deterministic
                #     )[0, :]
                #     print(action, _)
                # else:
                #     action = action
                #     print(action, _)
                # if _ % 10 == 0:
                #     action = old_actions[_//10]
                #     print(action, _)
                # else:
                #     action = action
                #     print(action, _)
                next_observation, reward, done, info = self.env.step(action)
                if dt_feat:
                    observation = np.hstack([
                        observation, [dt]]).astype(np.float32)
                observations.append(observation)
                # actions.append(action)
                actions.append(action*2)
                rewards.append(reward)
                dones.append(done)
                if 'score' in info:
                    successes.append(info['score'])
                else:
                    successes.append(0)
                next_observations.append(next_observation)
                if video and traj == 0:
                    if 'rgb_array' in self.env.metadata['render.modes']:
                        if True:
                            from d4rl.kitchen.adept_envs.franka.kitchen_multitask_v0 import KitchenTaskRelaxV1
                            imgs.append(KitchenTaskRelaxV1.render(self.env, 'rgb_array'))
                        else:
                            imgs.append(self.env.render(mode='rgb_array'))
                    else: # for metaworld
                        imgs.append(self.env.render(offscreen=True))


                if replay_buffer is not None:
                    replay_buffer.add_sample(
                        # observation, action, reward, next_observation, done
                        observation, action*2, reward, next_observation, done
                    )

                observation = next_observation

                if done:
                    break

            # import pickle
            # with open('actions.pkl','wb') as fp:
            #     pickle.dump(actions, fp)

            trajs.append(dict(
                observations=np.array(observations, dtype=np.float32),
                actions=np.array(actions, dtype=np.float32),
                rewards=np.array(rewards, dtype=np.float32),
                next_observations=np.array(next_observations, dtype=np.float32),
                dones=np.array(dones, dtype=np.float32),
                successes=np.array(successes, dtype=np.float32),
            ))
            if video and traj == 0:
                imgs = np.stack(imgs, axis=0)
                vid_from_frames(imgs, output_file)
                file_path_stem = os.path.splitext(output_file)[0]
                if qs:
                    q_estimates = []
                    for q in qs:
                        q_estimates.append(
                            q(torch.Tensor(observations).cuda(),
                            torch.Tensor(actions).cuda()).cpu().detach().numpy())
                    plot_q_over_traj(
                        q_estimates, rewards, imgs, f'{file_path_stem}_q.jpg')

        return trajs

    @property
    def env(self):
        return self._env
