# Copyright 2022 Spotify AB
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy

import numpy as np
import gym
import pybullet_envs
import os
import pickle
import argparse

from datetime import datetime
from d4rl_pybullet.sac import SAC, seed_everything
# from d4rl_sac_test import SAC, seed_everything
from utils import SimpleLogger
import matplotlib.pyplot as plt
from d4rl_pybullet.utility import save_buffer, save_hdf5


def update(buffer, sac, batch_size, train_actor=True):
    obs_ts = []
    act_ts = []
    rew_tp1s = []
    obs_tp1s = []
    ter_tp1s = []
    while len(obs_ts) != batch_size:
        index = np.random.randint(len(buffer) - 1)
        # skip if index indicates the terminal state
        if buffer[index][3][0]:
            continue
        obs_ts.append(buffer[index][0])
        act_ts.append(buffer[index][1])
        rew_tp1s.append(buffer[index + 1][2])
        obs_tp1s.append(buffer[index + 1][0])
        ter_tp1s.append(buffer[index + 1][3])

    critic_loss = sac.update_critic(obs_ts, act_ts, rew_tp1s, obs_tp1s, ter_tp1s)

    if train_actor:
        actor_loss = sac.update_actor(obs_ts)
    else:
        actor_loss = -1.0

    temp_loss = sac.update_temp(obs_ts)

    sac.update_target()

    return critic_loss, actor_loss, temp_loss


def update_with_bc(buffer, sac, batch_size, train_actor=True, exp_obs=None, exp_act=None, rng=None, alpha=0.5):
    obs_ts = []
    act_ts = []
    rew_tp1s = []
    obs_tp1s = []
    ter_tp1s = []
    while len(obs_ts) != batch_size:
        index = np.random.randint(len(buffer) - 1)
        # skip if index indicates the terminal state
        if buffer[index][3][0]:
            continue
        obs_ts.append(buffer[index][0])
        act_ts.append(buffer[index][1])
        rew_tp1s.append(buffer[index + 1][2])
        obs_tp1s.append(buffer[index + 1][0])
        ter_tp1s.append(buffer[index + 1][3])

    critic_loss = sac.update_critic(obs_ts, act_ts, rew_tp1s, obs_tp1s, ter_tp1s)

    if train_actor:
        actor_loss = sac.update_actor_with_reg(obs_ts, exp_obs_t=exp_obs, smallds_expert_a_t=exp_act, rng=rng, alpha=alpha)
    else:
        actor_loss = -1.0

    temp_loss = sac.update_temp(obs_ts)

    sac.update_target()

    return critic_loss, actor_loss, temp_loss


def evaluate(env, sac, n_episodes=10):
    episode_rews = []
    for episode in range(n_episodes):
        obs = env.reset()
        ter = False
        episode_rew = 0.0
        while not ter:
            act = sac.act([obs], deterministic=True)[0]
            obs, rew, ter, _ = env.step(act)
            episode_rew += rew
        episode_rews.append(episode_rew)
    return np.mean(episode_rews)


def save_expert_data(env, sac, n_episodes=16):
    dataset = {'actions': [], 'observations': [], 'rewards': [], 'terminals': []}
    episode_rews = []
    for episode in range(n_episodes):
        obs = env.reset()
        ter = False
        episode_rew = 0.0
        while not ter:
            act = sac.act([obs], deterministic=True)[0]
            obs, rew, ter, _ = env.step(act)
            episode_rew += rew
            dataset['actions'].append(act)
            dataset['observations'].append(obs)
            dataset['rewards'].append(rew)
            dataset['terminals'].append(ter)
        episode_rews.append(episode_rew)

    observations = np.array(dataset['observations'], dtype=np.float32)
    actions = np.array(dataset['actions'], dtype=np.float32)
    rewards = np.array(dataset['rewards'], dtype=np.float32)
    terminals = np.array(dataset['terminals'], dtype=np.float32)

    buffer_path = os.path.join('experts', f'{env.unwrapped.spec.name}_{(np.mean(episode_rews) // 100) * 100}_expert_data.hdf5')
    save_hdf5(observations, actions, rewards, terminals, buffer_path)

    print(f"Expert data is saved at the value of {np.mean(episode_rews)}.")
    return np.mean(episode_rews)


def train(
    env,
    eval_env,
    sac,
    logdir,
    desired_level,
    total_step,
    buffer=[],
    train_actor_threshold=None,
    batch_size=100,
    save_interval=10000,
    eval_interval=10000,
    recompute_reward=False,
    reward_recompute_interval=10000,
    eval2=False,
):
    logger = SimpleLogger(logdir)

    step = 0
    buffer_il = []
    true_reward_list = []
    while step <= total_step:
        obs_t = env.reset()
        ter_t = False
        rew_t = 0.0
        episode_rew = 0.0
        while not ter_t and step <= total_step:
            act_t = sac.act([obs_t])[0]

            buffer.append([obs_t, act_t, [rew_t], [ter_t]])
            buffer_il.append([obs_t, act_t, [rew_t], [ter_t]])

            obs_t, rew_t, ter_t, _ = env.step(act_t)

            episode_rew += rew_t

            if len(buffer) > batch_size:
                train_actor = (
                    step >= train_actor_threshold if train_actor_threshold else True
                )
                update(buffer, sac, batch_size, train_actor)
            # if step % save_interval == 0:
            #     sac.save(os.path.join(logdir, 'model_%d.pt' % step))

            if step % eval_interval == 0:
                eval_reward = evaluate(eval_env, sac)
                if eval2:
                    logger.add2(
                        "eval_reward",
                        step,
                        evaluate(eval_env, sac, n_episodes=100),
                        evaluate(env.clone(), sac, n_episodes=100),
                    )
                else:
                    logger.add("eval_reward", step, eval_reward)
                logger.plot("eval_reward", step, eval_reward)
                # if eval_reward > target_eval:
                #     break
            step += 1

            if recompute_reward and step % reward_recompute_interval == 0:
                env.recompute_reward(buffer_il)

        if ter_t:
            buffer.append([obs_t, np.zeros_like(act_t), [rew_t], [ter_t]])
            buffer_il.append([obs_t, np.zeros_like(act_t), [rew_t], [ter_t]])

        logger.add("reward", step, episode_rew)

        if desired_level is not None and episode_rew >= desired_level:
            break

    # save final buffer
    # save_buffer(buffer, logdir)
    # print('Final buffer has been saved.')

    # save final parameters
    sac.save(os.path.join(logdir, 'final_model.pt'))
    print('Final model parameters have been saved.')


def train_ilbc(
        env,
        eval_env,
        sac,
        logdir,
        desired_level,
        total_step,
        buffer=[],
        train_actor_threshold=None,
        batch_size=100,
        save_interval=10000,
        eval_interval=10000,
        recompute_reward=False,
        reward_recompute_interval=10000,
        eval2=False,
        exp_obs=None,
        exp_act=None,
        rng=None,
        alpha=0.5
):
    logger = SimpleLogger(logdir)

    step = 0
    buffer_il = []
    true_reward_list = []
    while step <= total_step:
        obs_t = env.reset()
        ter_t = False
        rew_t = 0.0
        episode_rew = 0.0
        while not ter_t and step <= total_step:
            act_t = sac.act([obs_t])[0]

            buffer.append([obs_t, act_t, [rew_t], [ter_t]])
            buffer_il.append([obs_t, act_t, [rew_t], [ter_t]])

            obs_t, rew_t, ter_t, _ = env.step(act_t)

            episode_rew += rew_t

            if len(buffer) > batch_size:
                train_actor = (
                    step >= train_actor_threshold if train_actor_threshold else True
                )
                update_with_bc(buffer, sac, batch_size, train_actor, exp_obs=exp_obs, exp_act=exp_act, rng=rng, alpha=alpha)
                if step < 5:
                    print(step, flush=True)
            # if step % save_interval == 0:
            #     sac.save(os.path.join(logdir, 'model_%d.pt' % step))

            if step % eval_interval == 0:
                if eval2:
                    logger.add2(
                        "eval_reward",
                        step,
                        evaluate(eval_env, sac, n_episodes=100),
                        evaluate(env.clone(), sac, n_episodes=100),
                    )
                else:
                    eval_result = evaluate(eval_env, sac)
                    print(step, eval_result, flush=True)
                    logger.add("eval_reward", step, eval_result)
                logger.plot("eval_reward", step, eval_result)

            step += 1

            if recompute_reward and step % reward_recompute_interval == 0:
                env.recompute_reward(buffer_il)

        if ter_t:
            buffer.append([obs_t, np.zeros_like(act_t), [rew_t], [ter_t]])
            buffer_il.append([obs_t, np.zeros_like(act_t), [rew_t], [ter_t]])

        logger.add("reward", step, episode_rew)

        if desired_level is not None and episode_rew >= desired_level:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--desired-level", type=float, default=400)
    parser.add_argument("--total-step", type=int, default=1000000)
    parser.add_argument("--gpu", type=int)
    args = parser.parse_args()

    env = gym.make('HalfCheetah', xml_file='/h/jiangm/opolo/il-by-rl-main/assets/halfcheetah.xml')
    eval_env = gym.make('HalfCheetah', xml_file='/h/jiangm/opolo/il-by-rl-main/assets/halfcheetah.xml')

    env.seed(args.seed)
    seed_everything(args.seed)

    observation_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    device = "cuda:%d" % args.gpu if args.gpu is not None else "cpu:0"

    sac = SAC(observation_size, action_size, device)

    logdir = os.path.join("logs", "{}_{}".format(args.env, args.seed))
    os.makedirs(logdir, exist_ok=True)
    if os.path.exists(os.path.join(logdir, f'final_model.pt')):
        sac.load(os.path.join(logdir, f'final_model.pt'))
    print(f"The initial agent is of level {evaluate(eval_env, sac)}", flush=True)

    train(env, eval_env, sac, logdir, args.desired_level, args.total_step)
    save_expert_data(eval_env, sac, n_episodes=16)
