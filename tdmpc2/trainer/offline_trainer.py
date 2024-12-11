import os
from copy import deepcopy
from time import time
from pathlib import Path
from glob import glob
import pickle

import numpy as np
import torch
from tqdm import tqdm
from tensordict.tensordict import TensorDict

from common.buffer import Buffer
from trainer.base import Trainer


class OfflineTrainer(Trainer):
    """Trainer class for multi-task offline TD-MPC2 training."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._start_time = time()
    
    def eval(self):
        """Evaluate a TD-MPC2 agent."""
        results = dict()
        for task_idx in tqdm(range(len(self.cfg.tasks)), desc='Evaluating'):
            ep_rewards, ep_successes = [], []
            for _ in range(self.cfg.eval_episodes):
                obs, done, ep_reward, t = self.env.reset(task_idx), False, 0, 0
                while not done:
                    action = self.agent.act(obs, t0=t==0, eval_mode=True, task=task_idx)
                    obs, reward, done, info = self.env.step(action)
                    ep_reward += reward
                    t += 1
                ep_rewards.append(ep_reward)
                ep_successes.append(info['success'])
            results.update({
                f'episode_reward+{self.cfg.tasks[task_idx]}': np.nanmean(ep_rewards),
                f'episode_success+{self.cfg.tasks[task_idx]}': np.nanmean(ep_successes),})
        return results
                
    def train(self):
        """Train a TD-MPC2 agent."""
        # assert self.cfg.multitask and self.cfg.task in {'mt30', 'mt80'}, \
        #     'Offline training only supports multitask training with mt30 or mt80 task sets.'

        # Load data
        # assert self.cfg.task in self.cfg.data_dir, \
        #     f'Expected data directory {self.cfg.data_dir} to contain {self.cfg.task}, ' \
        #     f'please double-check your config.'
        fp = Path(os.path.join(self.cfg.data_dir, '*.pt'))
        fps = sorted(glob(str(fp)))
        assert len(fps) > 0, f'No data found at {fp}'
        print(f'Found {len(fps)} files in {fp}')
    
        # Create buffer for sampling
        _cfg = deepcopy(self.cfg)
        _cfg.episode_length = 101 if self.cfg.task in ['mt80', 'mtgrab15'] else 501
        _cfg.buffer_size =  self.cfg.buffer_size #1200 # 550_450_000 if self.cfg.task == 'mt80' else 345_690_000
        _cfg.steps = _cfg.buffer_size
        self.buffer = Buffer(_cfg)
        for fp in tqdm(fps, desc='Loading data'):
            td = torch.load(fp)
            try:
                assert td.shape[1] == _cfg.episode_length, \
                f'Expected episode length {td.shape[1]} to match config episode length {_cfg.episode_length}, ' \
                f'please double-check your config.'
            except IndexError:
                td.shape = td['task'].shape
                assert td.shape[1] == _cfg.episode_length, \
                f'Expected episode length {td.shape[1]} to match config episode length {_cfg.episode_length}, ' \
                f'please double-check your config.'
            for i in range(len(td)):
                self.buffer.add(td[i])
        assert self.buffer.num_eps == self.buffer.capacity, \
            f'Buffer has {self.buffer.num_eps} episodes, expected {self.buffer.capacity} episodes.'
        
        print(f'Training agent for {self.cfg.steps} iterations...')
        metrics = {}
        for i in tqdm(range(self.cfg.steps), desc='Training'):

            # Update agent
            train_metrics = self.agent.update(self.buffer)

            # Evaluate agent periodically
            if i % self.cfg.eval_freq == 0 or i % 10_000 == 0 or i == self.cfg.steps-1:
                metrics = {
                    'iteration': i,
                    'total_time': time() - self._start_time,
                }
                metrics.update(train_metrics)
                if self.cfg.eval_enable and (i % self.cfg.eval_freq == 0 or i == self.cfg.steps-1):
                    metrics.update(self.eval())
                    self.logger.pprint_multitask(metrics, self.cfg)
                    if i > 0:
                        self.logger.save_agent(self.agent, identifier=f'{i}')
                self.logger.log(metrics, 'pretrain')
            
        self.logger.finish(self.agent)

    def train_transition(self):
        """Train a TD-MPC2 agent's transition model."""
        # assert self.cfg.multitask and self.cfg.task in {'mt30', 'mt80'}, \
        #     'Offline training only supports multitask training with mt30 or mt80 task sets.'

        # # Load data
        _cfg = deepcopy(self.cfg)
        _cfg.buffer_size = self.cfg.buffer_size
        _cfg.steps = _cfg.buffer_size
        self.buffer = Buffer(_cfg)  # 假设Buffer是你的数据缓冲区类
        
        fp = self.cfg.data_dir
        with open(fp, 'rb') as f:
            td = pickle.load(f)
        td = TensorDict({k: torch.tensor(v) for k, v in td.items()})
        # norm
        td['s'] = (td['s'] / 100.0) * 2 - 1     # [0,100] -> [-1, 1]
        td['s'][:, -1] = (td['s'][:, -1] + 1) / 7.0 - 1
        reward_mean = torch.mean(td['r'])
        reward_std = torch.std(td['r'])
        reward_dict = {'mean': reward_mean, 'std': reward_std}
        np.save('/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/chenxinyan-240108120066/chenxinyan/tdmpc2/visual/reward_params.npy', reward_dict)
        td['r'] = (td['r'] / reward_mean) / reward_std   # (miu=0, sigma=1)
        try:
            _cfg.episode_length = td.shape[1]
        except IndexError:
            td.shape = td['r'].shape
            _cfg.episode_length = td.shape[1]
        for i in range(len(td)):
            self.buffer.add(td[i])
        assert self.buffer.num_eps == self.buffer.capacity, \
            f'Buffer has {self.buffer.num_eps} episodes, expected {self.buffer.capacity} episodes.'
        
        # create eval buffer if eval
        if self.cfg.eval_enable:
            _cfg = deepcopy(self.cfg)
            _cfg.buffer_size = self.cfg.eval_buffer_size
            _cfg.steps = _cfg.buffer_size
            self.eval_buffer = Buffer(_cfg)  # 假设Buffer是你的数据缓冲区类
            
            fp = self.cfg.eval_data_dir
            with open(fp, 'rb') as f:
                td = pickle.load(f)
            td = TensorDict({k: torch.tensor(v) for k, v in td.items()})
            # norm
            td['s'] = (td['s'] / 100.0) * 2 - 1     # [0,100] -> [-1, 1]
            td['s'][:, -1] = (td['s'][:, -1] + 1) / 7.0 - 1
            td['r'] = (td['r'] / reward_mean) / reward_std   # (miu=0, sigma=1)
            try:
                _cfg.episode_length = td.shape[1]
            except IndexError:
                td.shape = td['r'].shape
                _cfg.episode_length = td.shape[1]
            for i in range(len(td)):
                self.eval_buffer.add(td[i])
            assert self.eval_buffer.num_eps == self.eval_buffer.capacity, \
                f'Buffer has {self.eval_buffer.num_eps} episodes, expected {self.eval_buffer.capacity} episodes.'
        
        print(f"Training agent's transition model for {self.cfg.steps} iterations...")
        metrics = {}
        for i in tqdm(range(self.cfg.steps), desc='Training Trans'):

            # Update agent
            train_metrics = self.agent.transition_update(self.buffer)

            # Evaluate agent periodically
            if i % self.cfg.eval_freq == 0 or i % 10_000 == 0 or i == self.cfg.steps-1:
                metrics = {
                    'iteration': i,
                    'total_time': time() - self._start_time,
                }
                metrics.update(train_metrics)
                if self.cfg.eval_enable and (i % self.cfg.eval_freq == 0 or i == self.cfg.steps-1):
                    metrics.update(self.agent.transition_eval(self.eval_buffer))
                    if i > 0:
                        self.logger.save_agent(self.agent, identifier=f'{i}')
                self.logger.log(metrics, 'pretrain')
            
        self.logger.finish(self.agent)


class MultiGPUOfflineTrainer(Trainer):
    """Trainer class for multi-task offline TD-MPC2 training."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._start_time = time()
    
    def eval(self):
        """Evaluate a TD-MPC2 agent."""
        results = dict()
        for task_idx in tqdm(range(len(self.cfg.tasks)), desc='Evaluating'):
            ep_rewards, ep_successes = [], []
            for _ in range(self.cfg.eval_episodes):
                obs, done, ep_reward, t = self.env.reset(task_idx), False, 0, 0
                while not done:
                    action = self.agent.act(obs, t0=t==0, eval_mode=True, task=task_idx)
                    obs, reward, done, info = self.env.step(action)
                    ep_reward += reward
                    t += 1
                ep_rewards.append(ep_reward)
                ep_successes.append(info['success'])
            results.update({
                f'episode_reward+{self.cfg.tasks[task_idx]}': np.nanmean(ep_rewards),
                f'episode_success+{self.cfg.tasks[task_idx]}': np.nanmean(ep_successes),})
        return results
                
    def train(self, accelerator):
        """Train a TD-MPC2 agent."""
        # assert self.cfg.multitask and self.cfg.task in {'mt30', 'mt80'}, \
        #     'Offline training only supports multitask training with mt30 or mt80 task sets.'

        # Load data
        # assert self.cfg.task in self.cfg.data_dir, \
        #     f'Expected data directory {self.cfg.data_dir} to contain {self.cfg.task}, ' \
        #     f'please double-check your config.'
        fp = Path(os.path.join(self.cfg.data_dir, '*.pt'))
        fps = sorted(glob(str(fp)))
        assert len(fps) > 0, f'No data found at {fp}'
        print(f'Found {len(fps)} files in {fp}')
    
        # Create buffer for sampling
        _cfg = deepcopy(self.cfg)
        _cfg.episode_length = 101 if self.cfg.task in ['mt80', 'mtgrab15'] else 501
        _cfg.buffer_size =  self.cfg.buffer_size #1200 # 550_450_000 if self.cfg.task == 'mt80' else 345_690_000
        _cfg.steps = _cfg.buffer_size
        self.buffer = Buffer(_cfg)
        for fp in tqdm(fps, desc='Loading data'):
            td = torch.load(fp)
            try:
                assert td.shape[1] == _cfg.episode_length, \
                f'Expected episode length {td.shape[1]} to match config episode length {_cfg.episode_length}, ' \
                f'please double-check your config.'
            except IndexError:
                td.shape = td['task'].shape
                assert td.shape[1] == _cfg.episode_length, \
                f'Expected episode length {td.shape[1]} to match config episode length {_cfg.episode_length}, ' \
                f'please double-check your config.'
            for i in range(len(td)):
                self.buffer.add(td[i])
        assert self.buffer.num_eps == self.buffer.capacity, \
            f'Buffer has {self.buffer.num_eps} episodes, expected {self.buffer.capacity} episodes.'
        
        print(f'Training agent for {self.cfg.steps} iterations...')
        metrics = {}
        for i in tqdm(range(self.cfg.steps), desc='Training'):

            # Update agent
            train_metrics = self.agent.update(self.buffer, accelerator)

            # Evaluate agent periodically
            if i % self.cfg.eval_freq == 0 or i % 10_000 == 0 or i == self.cfg.steps-1:
                metrics = {
                    'iteration': i,
                    'total_time': time() - self._start_time,
                }
                metrics.update(train_metrics)
                if i % self.cfg.eval_freq == 0 or i == self.cfg.steps-1:
                    if self.cfg.eval_enable:
                        metrics.update(self.eval())
                        self.logger.pprint_multitask(metrics, self.cfg)
                    if i > 0:
                        self.logger.save_agent(self.agent, identifier=f'{i}')
                self.logger.log(metrics, 'pretrain')
            
        self.logger.finish(self.agent)

