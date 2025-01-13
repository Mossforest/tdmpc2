from time import time

import numpy as np
import torch
from tensordict.tensordict import TensorDict

from trainer.base import Trainer


class OnlineTrainer(Trainer):
    """Trainer class for single-task online TD-MPC2 training."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._step = 0
        self._ep_idx = 0
        self._start_time = time()
        self.data_param_dict = np.load('/mnt/afs/chenxinyan/industrialbenchmark/industrial_benchmark_python/data/data_70_multisample_param_dict.npy', allow_pickle=True).item()

    def common_metrics(self):
        """Return a dictionary of current metrics."""
        return dict(
            step=self._step,
            episode=self._ep_idx,
            total_time=time() - self._start_time,
        )

    def eval(self):
        """Evaluate a TD-MPC2 agent."""
        ep_rewards, ep_successes = [], []
        for i in range(self.cfg.eval_episodes):
            obs, done, ep_reward, t = self.env.reset(), False, 0, 0
            # norm obs & reward
            for i in range(obs.shape[-1]):
                if i == 0:
                    obs[i] = obs[i] / 100.0 * 2 - 1  # norm -> [-1, 1]
                else:
                    i_mean = self.data_param_dict[f'{i}_mean']
                    i_std = self.data_param_dict[f'{i}_std']
                    obs[i] = (obs[i] - i_mean) / i_std   # (miu=0, sigma=1)
            reward_mean = self.data_param_dict['reward_mean']
            reward_std = self.data_param_dict['reward_std']
            
            if self.cfg.save_video:
                self.logger.video.init(self.env, enabled=(i==0))
            while not done:
                action = self.agent.act(obs, t0=t==0, eval_mode=True)
                obs, reward, done, info, next_states, rewards = self.env.step(action)
                
                # norm obs & reward
                for i in range(obs.shape[-1]):
                    if i == 0:
                        obs[i] = obs[i] / 100.0 * 2 - 1  # norm -> [-1, 1]
                    else:
                        i_mean = self.data_param_dict[f'{i}_mean']
                        i_std = self.data_param_dict[f'{i}_std']
                        obs[i] = (obs[i] - i_mean) / i_std   # (miu=0, sigma=1)
                reward = (reward - reward_mean) / reward_std
                ep_reward += reward
                t += 1
                if self.cfg.save_video:
                    self.logger.video.record(self.env)
            ep_rewards.append(ep_reward)
            ep_successes.append(info['success'])
            if self.cfg.save_video:
                self.logger.video.save(self._step)
        return dict(
            episode_reward=np.nanmean(ep_rewards),
            episode_success=np.nanmean(ep_successes),
        )

    def to_td(self, obs, action=None, reward=None, next_states=None, rewards=None):
        """Creates a TensorDict for a new episode."""
        if isinstance(obs, dict):
            obs = TensorDict(obs, batch_size=(), device='cpu')
        else:
            obs = obs.unsqueeze(0).cpu()
        if action is None:
            action = torch.full_like(self.env.rand_act(), float('nan'))
        if reward is None:
            reward = torch.tensor(float('nan'))
        if next_states is None:
            next_states = torch.zeros((5, obs.shape[-1]), dtype=obs.dtype)
        if rewards is None:
            rewards = torch.zeros(5, dtype=torch.float)
        td = TensorDict(dict(
            s=obs,
            a=action.unsqueeze(0),
            r=reward.unsqueeze(0),
            next_s_samples=next_states.unsqueeze(0),
            r_samples=rewards.unsqueeze(0),
        ), batch_size=(1,))
        return td

    def train(self):
        """Train a TD-MPC2 agent."""
        train_metrics, done, eval_next = {}, True, False
        while self._step <= self.cfg.steps:

            # Evaluate agent periodically
            if self._step > 0 and self._step % self.cfg.eval_freq == 0:
                eval_next = True

            # Reset environment
            if done:
                if eval_next:
                    eval_metrics = self.eval()
                    eval_metrics.update(self.common_metrics())
                    self.logger.log(eval_metrics, 'eval')
                    eval_next = False

                if self._step > 0:
                    train_metrics.update(
                        episode_reward=torch.tensor([td['r'] for td in self._tds[1:]]).sum(),
                        episode_success=info['success'],
                    )
                    train_metrics.update(self.common_metrics())
                    self.logger.log(train_metrics, 'train')
                    self._ep_idx = self.buffer.add(torch.cat(self._tds))

                obs = self.env.reset()
                # norm obs & reward
                for i in range(obs.shape[-1]):
                    if i == 0:
                        obs[i] = obs[i] / 100.0 * 2 - 1
                    else:
                        i_mean = self.data_param_dict[f'{i}_mean']
                        i_std = self.data_param_dict[f'{i}_std']
                        obs[i] = (obs[i] - i_mean) / i_std
                reward_mean = self.data_param_dict['reward_mean']
                reward_std = self.data_param_dict['reward_std']
                self._tds = [self.to_td(obs)]

            # Collect experience
            if self._step > self.cfg.seed_steps:
                action = self.agent.act(obs, t0=len(self._tds)==1)
            else:
                action = self.env.rand_act()
            obs, reward, done, info, next_states, rewards = self.env.step(action)
            # norm obs & reward
            for i in range(obs.shape[-1]):
                if i == 0:
                    obs[i] = obs[i] / 100.0 * 2 - 1
                    next_states[:, i] = (next_states[:, i] / 100.0) * 2 - 1
                else:
                    i_mean = self.data_param_dict[f'{i}_mean']
                    i_std = self.data_param_dict[f'{i}_std']
                    obs[i] = (obs[i] - i_mean) / i_std
                    next_states[:, i] = (next_states[:, i] - i_mean) / i_std
            reward_mean = self.data_param_dict['reward_mean']
            reward_std = self.data_param_dict['reward_std']
            reward = (reward - reward_mean) / reward_std
            rewards = (rewards - reward_mean) / reward_std
            self._tds.append(self.to_td(obs, action, reward, next_states, rewards))

            # Update agent
            if self._step >= self.cfg.seed_steps:
                if self._step == self.cfg.seed_steps:
                    num_updates = self.cfg.seed_steps
                    print('Pretraining agent on seed data...')
                else:
                    num_updates = 1
                for _ in range(num_updates):
                    _train_metrics = self.agent.update(self.buffer)
                train_metrics.update(_train_metrics)

            self._step += 1
    
        self.logger.finish(self.agent)

class MultiGPUOnlineTrainer(Trainer):
    """Trainer class for single-task online TD-MPC2 training."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._step = 0
        self._ep_idx = 0
        self._start_time = time()

    def common_metrics(self):
        """Return a dictionary of current metrics."""
        return dict(
            step=self._step,
            episode=self._ep_idx,
            total_time=time() - self._start_time,
        )

    def eval(self):
        """Evaluate a TD-MPC2 agent."""
        ep_rewards, ep_successes = [], []
        for i in range(self.cfg.eval_episodes):
            obs, done, ep_reward, t = self.env.reset(), False, 0, 0
            if self.cfg.save_video:
                self.logger.video.init(self.env, enabled=(i==0))
            while not done:
                action = self.agent.act(obs, t0=t==0, eval_mode=True)
                obs, reward, done, info = self.env.step(action)
                ep_reward += reward
                t += 1
                if self.cfg.save_video:
                    self.logger.video.record(self.env)
            ep_rewards.append(ep_reward)
            ep_successes.append(info['success'])
            if self.cfg.save_video:
                self.logger.video.save(self._step)
        return dict(
            episode_reward=np.nanmean(ep_rewards),
            episode_success=np.nanmean(ep_successes),
        )

    def to_td(self, obs, action=None, reward=None):
        """Creates a TensorDict for a new episode."""
        if isinstance(obs, dict):
            obs = TensorDict(obs, batch_size=(), device='cpu')
        else:
            obs = obs.unsqueeze(0).cpu()
        if action is None:
            action = torch.full_like(self.env.rand_act(), float('nan'))
        if reward is None:
            reward = torch.tensor(float('nan'))
        td = TensorDict(dict(
            obs=obs,
            action=action.unsqueeze(0),
            reward=reward.unsqueeze(0),
        ), batch_size=(1,))
        return td

    def train(self, accelerator):
        """Train a TD-MPC2 agent."""

        train_metrics, done, eval_next = {}, True, False
        while self._step <= self.cfg.steps:

            # Evaluate agent periodically
            if self._step > 0 and self._step % self.cfg.eval_freq == 0:
                eval_next = True

            # Reset environment
            if done:
                if eval_next:
                    accelerator.print("Begin evaluation")
                    eval_metrics = self.eval()
                    eval_metrics.update(self.common_metrics())
                    self.logger.log(eval_metrics, 'eval')
                    eval_next = False
                    accelerator.print("End evaluation")

                if self._step > 0:
                    train_metrics.update(
                        episode_reward=torch.tensor([td['r'] for td in self._tds[1:]]).sum(),
                        episode_success=info['success'],
                    )
                    train_metrics.update(self.common_metrics())
                    self.logger.log(train_metrics, 'train')
                    self._ep_idx = self.buffer.add(torch.cat(self._tds))

                obs = self.env.reset()
                self._tds = [self.to_td(obs)]

            # Collect experience
            accelerator.print("Begin collecting experience")
            if self._step > self.cfg.seed_steps:
                action = self.agent.act(obs, t0=len(self._tds)==1)
            else:
                action = self.env.rand_act()
            obs, reward, done, info = self.env.step(action)
            self._tds.append(self.to_td(obs, action, reward))
            accelerator.print("End collecting experience")

            # Update agent
            if self._step >= self.cfg.seed_steps:
                accelerator.print("Begin updating agent")
                if self._step == self.cfg.seed_steps:
                    num_updates = self.cfg.seed_steps
                    print('Pretraining agent on seed data...')
                else:
                    num_updates = 1
                for _ in range(num_updates):
                    _train_metrics = self.agent.update(self.buffer, accelerator)
                train_metrics.update(_train_metrics)
                accelerator.print("End updating agent")

            self._step += 1
    
        self.logger.finish(self.agent)

