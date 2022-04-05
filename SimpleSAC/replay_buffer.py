from copy import copy, deepcopy
import h5py
from queue import Queue
import threading

import d4rl

import numpy as np
import torch


class ReplayBuffer(object):
    def __init__(self, max_size, data=None):
        self._max_size = max_size
        self._next_idx = 0
        self._size = 0
        self._initialized = False
        self._total_steps = 0

        if data is not None:
            if self._max_size < data['observations'].shape[0]:
                self._max_size = data['observations'].shape[0]
            self.add_batch(data)

    def __len__(self):
        return self._size

    def _init_storage(self, observation_dim, action_dim):
        self._observation_dim = observation_dim
        self._action_dim = action_dim
        self._observations = np.zeros((self._max_size, observation_dim), dtype=np.float32)
        self._next_observations = np.zeros((self._max_size, observation_dim), dtype=np.float32)
        self._actions = np.zeros((self._max_size, action_dim), dtype=np.float32)
        self._rewards = np.zeros(self._max_size, dtype=np.float32)
        self._dones = np.zeros(self._max_size, dtype=np.float32)
        self._next_idx = 0
        self._size = 0
        self._initialized = True

    def add_sample(self, observation, action, reward, next_observation, done):
        if not self._initialized:
            self._init_storage(observation.size, action.size)

        self._observations[self._next_idx, :] = np.array(observation, dtype=np.float32)
        self._next_observations[self._next_idx, :] = np.array(next_observation, dtype=np.float32)
        self._actions[self._next_idx, :] = np.array(action, dtype=np.float32)
        self._rewards[self._next_idx] = reward
        self._dones[self._next_idx] = float(done)

        if self._size < self._max_size:
            self._size += 1
        self._next_idx = (self._next_idx + 1) % self._max_size
        self._total_steps += 1

    def add_traj(self, observations, actions, rewards, next_observations, dones):
        for o, a, r, no, d in zip(observations, actions, rewards, next_observations, dones):
            self.add_sample(o, a, r, no, d)

    def add_batch(self, batch):
        self.add_traj(
            batch['observations'], batch['actions'], batch['rewards'],
            batch['next_observations'], batch['dones']
        )

    def sample(self, batch_size):
        indices = np.random.randint(len(self), size=batch_size)
        return self.select(indices)

    def select(self, indices):
        return dict(
            observations=self._observations[indices, ...],
            actions=self._actions[indices, ...],
            rewards=self._rewards[indices, ...],
            next_observations=self._next_observations[indices, ...],
            dones=self._dones[indices, ...],
        )

    def store(self, h5path):
        """Stores buffer data as an h5py file."""
        dataset_file = h5py.File(h5path, "w")
        dataset_file.create_dataset("obs", data=self._observations)
        dataset_file.create_dataset("actions", data=self._actions)
        dataset_file.create_dataset("next_obs", data=self._next_observations)
        dataset_file.create_dataset("rewards", data=self._rewards)
        dataset_file.create_dataset("dones", data=self._dones)
        dataset_file.close()
    
    def generator(self, batch_size, n_batchs=None):
        i = 0
        while n_batchs is None or i < n_batchs:
            yield self.sample(batch_size)
            i += 1

    @property
    def total_steps(self):
        return self._total_steps

    @property
    def data(self):
        return dict(
            observations=self._observations[:self._size, ...],
            actions=self._actions[:self._size, ...],
            rewards=self._rewards[:self._size, ...],
            next_observations=self._next_observations[:self._size, ...],
            dones=self._dones[:self._size, ...]
        )


def batch_to_torch(batch, device):
    return {
        k: torch.from_numpy(v).to(device=device, non_blocking=True)
        for k, v in batch.items()
    }


def load_d4rl_dataset(env):
    dataset = d4rl.qlearning_dataset(env)
    return dict(
        observations=dataset['observations'],
        actions=dataset['actions'],
        next_observations=dataset['next_observations'],
        rewards=dataset['rewards'],
        dones=dataset['terminals'].astype(np.float32),
    )

def load_dataset(h5path):
    dataset_file = h5py.File(h5path, "r")
    dataset = dict(
        observations=dataset_file["obs"][:].astype(np.float32),
        actions=dataset_file["actions"][:].astype(np.float32),
        next_observations=dataset_file["next_obs"][:].astype(np.float32),
        rewards=dataset_file["rewards"][:].astype(np.float32),
        dones=dataset_file["dones"][:].astype(np.float32),
    )
    # TODO: make consistent
    for k, v in dataset.items():
        if len(v.shape) > 1:
            dim_obs = v.shape[1]
        else:
            dim_obs = 1
        # v = v[-250000:] # all of the mujoco buffers are empty after 500k
        # dataset[k] = v.reshape(500, 500, dim_obs) #v.reshape(500, 500, dim_obs) #v.reshape(500, 320, dim_obs) #v.reshape(500, 1000, dim_obs) # this only works for mujoco
        dataset[k] = v.reshape(-1, 256, dim_obs) # this only works for pendulum
    return dataset

def index_batch(batch, indices):
    indexed = {}
    for key in batch.keys():
        indexed[key] = batch[key][indices, ...]
    return indexed

def index_batch_flat_n(batch, indices, size, n_steps):
    """Index into batches formatted (B, D)"""
    indexed = {}
    for key in batch.keys():
        indexed[key] = batch[key][indices].reshape(size, int(n_steps), -1)
    return indexed

def index_batch_n(batch, indices, size, n_steps):
    """Index into batches formatted (T, B, D)"""
    indexed = {}
    for key in batch.keys():
        indexed[key] = batch[key][indices[0], indices[1]].reshape(size, int(n_steps), -1)
    return indexed

def parition_batch_train_test(batch, train_ratio):
    train_indices = np.random.rand(batch['observations'].shape[0]) < train_ratio
    train_batch = index_batch(batch, train_indices)
    test_batch = index_batch(batch, ~train_indices)
    return train_batch, test_batch

def subsample_batch(batch, size):
    indices = np.random.randint(batch['observations'].shape[0], size=size)
    return index_batch(batch, indices)

def subsample_flat_batch_n(batch, size, n_steps):
    dones = batch['dones'].nonzero()[0]
    # concatenate done_idx with traj lens
    dones_and_lens = np.vstack((
        np.roll(dones, shift=1),
        np.diff(dones, prepend=0)))
    dones_and_lens[0,0] = 0
    # select (batch_idx, len) pairs from batch
    batch_indices = np.random.choice(dones.shape[0], size=size)
    batch_indices_and_lens = dones_and_lens[:,batch_indices]
    # select steps from trajectory
    traj_indices = np.random.randint(batch_indices_and_lens[1]-n_steps)
    # add with done_idx to get flat indices
    indices = traj_indices + batch_indices_and_lens[0]
    # add next n_steps to trajectory indices
    ascending_idxs = np.tile(np.arange(n_steps), size)
    indices = np.repeat(indices, n_steps) + ascending_idxs
    return index_batch_flat_n(batch, indices, size, n_steps) # B, N, D

def subsample_batch_n(batch, size, n_steps):
    ascending_idxs = np.tile(np.arange(n_steps), size)
    # pick random steps in trajectory
    traj_indices = np.random.randint(batch['rewards'].shape[0]-n_steps, size=size)
    # add next n_steps to trajectory indices
    traj_indices = np.repeat(traj_indices, n_steps) + ascending_idxs
    # pick trajectories from batch and repeat for n steps
    batch_indices = np.random.randint(batch['rewards'].shape[1], size=size)
    batch_indices = np.repeat(batch_indices, n_steps)
    indices = np.vstack((traj_indices, batch_indices)).astype(int) # should be T, B
    return index_batch_n(batch, indices, size, n_steps) # N, 1, D


def concatenate_batches(batches):
    concatenated = {}
    for key in batches[0].keys():
        concatenated[key] = np.concatenate([batch[key] for batch in batches], axis=0).astype(np.float32)
    return concatenated


def split_batch(batch, batch_size):
    batches = []
    length = batch['observations'].shape[0]
    keys = batch.keys()
    for start in range(0, length, batch_size):
        end = min(start + batch_size, length)
        batches.append({key: batch[key][start:end, ...] for key in keys})
    return batches


def split_data_by_traj(data, max_traj_length):
    dones = data['dones'].astype(bool)
    start = 0
    splits = []
    for i, done in enumerate(dones):
        if i - start + 1 >= max_traj_length or done:
            splits.append(index_batch(data, slice(start, i + 1)))
            start = i + 1

    if start < len(dones):
        splits.append(index_batch(data, slice(start, None)))

    return splits
