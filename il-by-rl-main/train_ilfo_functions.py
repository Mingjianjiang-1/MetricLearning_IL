import copy

import gym

import torch
import numpy as np
from gym.core import Wrapper

import os

from utils import prepare_run

from d4rl_sac import SAC
from d4rl_train import train
from sklearn.neighbors import NearestNeighbors


class ILFORewardWrapper(Wrapper):
    def __init__(self, env, env_name, is_in_support):
        super().__init__(env)
        self.is_in_support = is_in_support
        self.env_name = env_name
        self.prev_state = None

    def step(self, action):
        env_state, orig_reward, is_terminal, d = self.env.step(action)
        assert self.prev_state is not None
        il_reward = self.is_in_support(
            np.hstack([self.prev_state, env_state])
        )
        self.prev_state = env_state
        return env_state, il_reward, is_terminal, d

    def reset(self, **kwargs):
        self.prev_state = self.env.reset(**kwargs)
        return self.prev_state

    def clone(self):
        new_env = gym.make(self.env_name)
        new_env = ILFORewardWrapper(
            new_env,
            self.env_name,
            copy.deepcopy(self.is_in_support),
        )
        new_env.prev_state = copy.deepcopy(self.prev_state)
        return new_env


def make_buffer_next(buffer):
    buffer_next = copy.deepcopy(buffer)
    buffer_next.pop(0)
    return buffer_next


def train_ilfo_actor(
        no_episodes,
        env_name,
        file_path,
        h5path,
        action_multiplier=1.0,
        gamma=0.99,
        initial_actor_path=None,
        eval2=False,
        eval_interval=10000,
        seed=0,
        total_steps=500000,
):
    buffer, env, eval_env, smallds_expert_a_t, smallds_obs_t = prepare_run(
        env_name, file_path, h5path, no_episodes, seed
    )

    observation_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]

    is_in_support = get_binary_reward(smallds_obs_t)

    w_env = ILFORewardWrapper(env, env_name, is_in_support)

    rl_algorithm = SAC(observation_size, action_size, "cuda", gamma=gamma)

    if initial_actor_path is not None:
        rl_algorithm.actor.load_state_dict(torch.load(initial_actor_path))
        rl_algorithm.actor.eval()

    sac_directory = file_path
    if not os.path.exists(sac_directory):
        os.makedirs(sac_directory)
    train(
        w_env,
        eval_env,
        rl_algorithm,
        sac_directory,
        None,
        total_steps,
        buffer=buffer,
        eval2=eval2,
        eval_interval=eval_interval,
    )

    # torch.save(rl_algorithm.actor.state_dict(), file_path)
    return rl_algorithm.actor


def get_binary_reward(smallds_obs_t):
    smallds_obs_nextobs_t = np.hstack(
        [smallds_obs_t[:-1], smallds_obs_t[1:]]
    )
    is_in_support = support_estimator(smallds_obs_nextobs_t)
    assert is_in_support(smallds_obs_nextobs_t[0, :]) > 0.0
    return is_in_support


def support_estimator(points):
    tolerance = 0.0
    nbrs1 = NearestNeighbors(n_neighbors=1, algorithm="ball_tree").fit(points)

    def is_in_support(x):
        distance, _ = nbrs1.kneighbors(x.reshape((1, len(x))))
        distance = float(distance)
        if distance <= tolerance:
            return 1.0
        how_far = distance - tolerance  # > 0
        return 1.0 - how_far**2  # was 1.0 - how_far ** 2

    return is_in_support
