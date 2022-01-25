import numpy as np


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
            )[0, :]
            next_observation, reward, done, _ = self.env.step(action)
            observations.append(observation)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            next_observations.append(next_observation)

            if replay_buffer is not None:
                replay_buffer.add_sample(
                    observation, action, reward, next_observation, done
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

    def sample(self, policy, n_trajs, deterministic=False, replay_buffer=None, video=False):
        trajs = []
        for _ in range(n_trajs):
            observations = []
            actions = []
            rewards = []
            next_observations = []
            dones = []
            if video:
                imgs = []

            observation = self.env.reset()

            # if you want to play back actions at different dt, uncomment
            # import pickle
            # old_actions = pickle.load(open('actions.pkl', 'rb'))
            for _ in range(self.max_traj_length):
                # for dt conditioned policy
                observation = np.hstack([
                    observation, [self._env.dt]]).astype(np.float32)
                action = policy(
                    np.expand_dims(observation, 0), deterministic=deterministic
                )[0, :]
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
                next_observation, reward, done, _ = self.env.step(action)
                observations.append(observation)
                actions.append(action)
                rewards.append(reward)
                dones.append(done)
                next_observations.append(next_observation)
                if video:
                    imgs.append(self.env.render(mode='rgb_array'))

                if replay_buffer is not None:
                    replay_buffer.add_sample(
                        observation, action, reward, next_observation, done
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
            ))
            if video:
                imgs = np.stack(imgs, axis=0)
                np.savez('movie.npz', imgs)

        return trajs

    @property
    def env(self):
        return self._env
