import numpy as np
import torch
import torch.nn.functional as F
from tensordict import TensorDict
from termcolor import colored

from common import math
from common.scale import RunningScale
from common.world_model import WorldModel, WorldModel_Flow
from common.optim import configure_weight_decay


class TDMPC2:
    """
    TD-MPC2 agent. Implements training + inference.
    Can be used for both single-task and multi-task experiments,
    and supports both state and pixel observations.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device('cuda')
        self.model = WorldModel(cfg).to(self.device)
        if cfg.pretrained_path:
            self.load(cfg.pretrained_path)
            print(f'loaded pretrained model from', colored(cfg.pretrained_path, 'yellow', attrs=['bold']))
        else:
            print(colored('train from scratch', 'yellow', attrs=['bold']))
        # wd list
        dynamic_params_1, dynamic_params_2 = configure_weight_decay(self.model._dynamics, self.cfg.weight_decay)
        self.optim = torch.optim.Adam([
            {'params': self.model._encoder.parameters(), 'lr': self.cfg.lr*self.cfg.enc_lr_scale},
            dynamic_params_1, dynamic_params_2,
            {'params': self.model._reward.parameters()},
            {'params': self.model._Qs.parameters()},
            {'params': self.model._task_emb.parameters() if self.cfg.multitask else []}
        ], lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        self.pi_optim = torch.optim.Adam(self.model._pi.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay, eps=1e-5)
        self.model.eval()
        self.scale = RunningScale(cfg)
        self.cfg.iterations += 2*int(cfg.action_dim >= 20) # Heuristic for large action spaces
        self.discount = torch.tensor(
            [self._get_discount(ep_len) for ep_len in cfg.episode_lengths], device='cuda'
        ) if self.cfg.multitask else self._get_discount(cfg.episode_length)

    def _get_discount(self, episode_length):
        """
        Returns discount factor for a given episode length.
        Simple heuristic that scales discount linearly with episode length.
        Default values should work well for most tasks, but can be changed as needed.

        Args:
            episode_length (int): Length of the episode. Assumes episodes are of fixed length.

        Returns:
            float: Discount factor for the task.
        """
        frac = episode_length/self.cfg.discount_denom
        return min(max((frac-1)/(frac), self.cfg.discount_min), self.cfg.discount_max)

    def save(self, fp):
        """
        Save state dict of the agent to filepath.
        
        Args:
            fp (str): Filepath to save state dict to.
        """
        torch.save({"model": self.model.state_dict()}, fp)

    def load(self, fp):
        """
        Load a saved state dict from filepath (or dictionary) into current agent.
        
        Args:
            fp (str or dict): Filepath or state dict to load.
        """
        state_dict = fp if isinstance(fp, dict) else torch.load(fp)
        self.model.load_state_dict(state_dict["model"])

    @torch.no_grad()
    def act(self, obs, t0=False, eval_mode=False, task=None):
        """
        Select an action by planning in the latent space of the world model.
        
        Args:
            obs (torch.Tensor): Observation from the environment.
            t0 (bool): Whether this is the first observation in the episode.
            eval_mode (bool): Whether to use the mean of the action distribution.
            task (int): Task index (only used for multi-task experiments).
        
        Returns:
            torch.Tensor: Action to take in the environment.
        """
        obs = obs.to(self.device, non_blocking=True).unsqueeze(0)
        if task is not None:
            task = torch.tensor([task], device=self.device)
        z = self.model.encode(obs, task)
        if self.cfg.mpc:
            a = self.plan(z, t0=t0, eval_mode=eval_mode, task=task)
        else:
            a = self.model.pi(z, task)[int(not eval_mode)][0]
        return a.cpu()

    @torch.no_grad()
    def _estimate_value(self, z, actions, task):
        """Estimate value of a trajectory starting at latent state z and executing given actions."""
        G, discount = 0, 1
        for t in range(self.cfg.horizon):
            reward = math.two_hot_inv(self.model.reward(z, actions[t], task), self.cfg)
            z = self.model.next(z, actions[t], task)
            G += discount * reward
            discount *= self.discount[torch.tensor(task)] if self.cfg.multitask else self.discount
        return G + discount * self.model.Q(z, self.model.pi(z, task)[1], task, return_type='avg')

    @torch.no_grad()
    def plan(self, z, t0=False, eval_mode=False, task=None):
        """
        Plan a sequence of actions using the learned world model.
        
        Args:
            z (torch.Tensor): Latent state from which to plan.
            t0 (bool): Whether this is the first observation in the episode.
            eval_mode (bool): Whether to use the mean of the action distribution.
            task (Torch.Tensor): Task index (only used for multi-task experiments).

        Returns:
            torch.Tensor: Action to take in the environment.
        """        
        # Sample policy trajectories
        if self.cfg.num_pi_trajs > 0:
            pi_actions = torch.empty(self.cfg.horizon, self.cfg.num_pi_trajs, self.cfg.action_dim, device=self.device)
            _z = z.repeat(self.cfg.num_pi_trajs, 1)
            for t in range(self.cfg.horizon-1):
                pi_actions[t] = self.model.pi(_z, task)[1]
                _z = self.model.next(_z, pi_actions[t], task)
            pi_actions[-1] = self.model.pi(_z, task)[1]

        # Initialize state and parameters
        z = z.repeat(self.cfg.num_samples, 1)
        mean = torch.zeros(self.cfg.horizon, self.cfg.action_dim, device=self.device)
        std = self.cfg.max_std*torch.ones(self.cfg.horizon, self.cfg.action_dim, device=self.device)
        if not t0:
            mean[:-1] = self._prev_mean[1:]
        actions = torch.empty(self.cfg.horizon, self.cfg.num_samples, self.cfg.action_dim, device=self.device)
        if self.cfg.num_pi_trajs > 0:
            actions[:, :self.cfg.num_pi_trajs] = pi_actions
    
        # Iterate MPPI
        for _ in range(self.cfg.iterations):

            # Sample actions
            actions[:, self.cfg.num_pi_trajs:] = (mean.unsqueeze(1) + std.unsqueeze(1) * \
                torch.randn(self.cfg.horizon, self.cfg.num_samples-self.cfg.num_pi_trajs, self.cfg.action_dim, device=std.device)) \
                .clamp(-1, 1)
            if self.cfg.multitask:
                actions = actions * self.model._action_masks[task]

            # Compute elite actions
            value = self._estimate_value(z, actions, task).nan_to_num_(0)
            elite_idxs = torch.topk(value.squeeze(1), self.cfg.num_elites, dim=0).indices
            elite_value, elite_actions = value[elite_idxs], actions[:, elite_idxs]

            # Update parameters
            max_value = elite_value.max(0)[0]
            score = torch.exp(self.cfg.temperature*(elite_value - max_value))
            score /= score.sum(0)
            mean = torch.sum(score.unsqueeze(0) * elite_actions, dim=1) / (score.sum(0) + 1e-9)
            std = torch.sqrt(torch.sum(score.unsqueeze(0) * (elite_actions - mean.unsqueeze(1)) ** 2, dim=1) / (score.sum(0) + 1e-9)) \
                .clamp_(self.cfg.min_std, self.cfg.max_std)
            if self.cfg.multitask:
                mean = mean * self.model._action_masks[task]
                std = std * self.model._action_masks[task]

        # Select action
        score = score.squeeze(1).cpu().numpy()
        actions = elite_actions[:, np.random.choice(np.arange(score.shape[0]), p=score)]
        self._prev_mean = mean
        a, std = actions[0], std[0]
        if not eval_mode:
            a += std * torch.randn(self.cfg.action_dim, device=std.device)
        return a.clamp_(-1, 1)
        
    def update_pi(self, zs, task):
        """
        Update policy using a sequence of latent states.
        
        Args:
            zs (torch.Tensor): Sequence of latent states.
            task (torch.Tensor): Task index (only used for multi-task experiments).

        Returns:
            float: Loss of the policy update.
        """
        self.pi_optim.zero_grad(set_to_none=True)
        self.model.track_q_grad(False)
        _, pis, log_pis, _ = self.model.pi(zs, task)
        qs = self.model.Q(zs, pis, task, return_type='avg')
        self.scale.update(qs[0])
        qs = self.scale(qs)

        # Loss is a weighted sum of Q-values
        rho = torch.pow(self.cfg.rho, torch.arange(len(qs), device=self.device))
        pi_loss = ((self.cfg.entropy_coef * log_pis - qs).mean(dim=(1,2)) * rho).mean()
        pi_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model._pi.parameters(), self.cfg.grad_clip_norm)
        self.pi_optim.step()
        self.model.track_q_grad(True)

        return pi_loss.item()

    @torch.no_grad()
    def _td_target(self, next_z, reward, task):
        """
        Compute the TD-target from a reward and the observation at the following time step.
        
        Args:
            next_z (torch.Tensor): Latent state at the following time step.
            reward (torch.Tensor): Reward at the current time step.
            task (torch.Tensor): Task index (only used for multi-task experiments).
        
        Returns:
            torch.Tensor: TD-target.
        """
        pi = self.model.pi(next_z, task)[1]
        discount = self.discount[task].unsqueeze(-1) if self.cfg.multitask else self.discount
        return reward + discount * self.model.Q(next_z, pi, task, return_type='min', target=True)

    def update(self, buffer):
        """
        Main update function. Corresponds to one iteration of model learning.
        
        Args:
            buffer (common.buffer.Buffer): Replay buffer.
        
        Returns:
            dict: Dictionary of training statistics.
        """
        # TODO: task??
        obs, action, reward, next_s_samples, r_samples, task = buffer.sample()
        obs = obs.float()
        action = action.float()
        next_s_samples = next_s_samples.float()
    
        # Compute targets
        with torch.no_grad():
            next_z = self.model.encode(obs[1:], task)
            td_targets = self._td_target(next_z, reward, task)
            
            # multiple next_z samples
            next_z_samples = torch.empty(self.cfg.horizon, self.cfg.batch_size, next_s_samples.shape[2], self.cfg.latent_dim, device=self.device)
            for i in range(next_s_samples.shape[2]):
                next_z_samples[:, :, i, :] = self.model.encode(next_s_samples[:, :, i, :], task)

        # Prepare for update
        self.optim.zero_grad(set_to_none=True)
        self.model.train()

        # Latent rollout
        zs = torch.empty(self.cfg.horizon+1, self.cfg.batch_size, self.cfg.latent_dim, device=self.device)
        z = self.model.encode(obs[0], task)
        zs[0] = z
        consistency_loss = 0
        distribution_loss = 0
        first_distribution_loss = None
        for t in range(self.cfg.horizon):
            z = self.model.next(z, action[t], task)
            consistency_loss += F.mse_loss(z, next_z[t]) * self.cfg.rho**t
            repeated_z = z.unsqueeze(1).repeat(1, next_z_samples.shape[2], 1)
            distribution_loss += F.kl_div(
                F.log_softmax(repeated_z, dim=2),
                F.softmax(next_z_samples[t], dim=2),
                reduction='batchmean'
            ) * self.cfg.rho**t
            if first_distribution_loss is None:
                first_distribution_loss = distribution_loss
            zs[t+1] = z

        # Predictions
        _zs = zs[:-1]
        qs = self.model.Q(_zs, action, task, return_type='all')
        reward_preds = self.model.reward(_zs, action, task)
        
        # Compute losses
        reward_loss, value_loss = 0, 0
        for t in range(self.cfg.horizon):
            reward_loss += math.soft_ce(reward_preds[t], reward[t], self.cfg).mean() * self.cfg.rho**t
            for q in range(self.cfg.num_q):
                value_loss += math.soft_ce(qs[q][t], td_targets[t], self.cfg).mean() * self.cfg.rho**t
        consistency_loss *= (1/self.cfg.horizon)
        distribution_loss *= (1/self.cfg.horizon)
        reward_loss *= (1/self.cfg.horizon)
        value_loss *= (1/(self.cfg.horizon * self.cfg.num_q))
        total_loss = (
            self.cfg.consistency_coef * consistency_loss +
            self.cfg.reward_coef * reward_loss +
            self.cfg.value_coef * value_loss
        )

        # Update model
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
        self.optim.step()

        # Update policy
        pi_loss = self.update_pi(zs.detach(), task)

        # Update target Q-functions
        self.model.soft_update_target_Q()

        # Return training statistics
        self.model.eval()
        return {
            "consistency_loss": float(consistency_loss.mean().item()),
            "reward_loss": float(reward_loss.mean().item()),
            "value_loss": float(value_loss.mean().item()),
            "pi_loss": pi_loss,
            "total_loss": float(total_loss.mean().item()),
            "grad_norm": float(grad_norm),
            "pi_scale": float(self.scale.value),
            "first_distribution_loss": float(first_distribution_loss.mean().item()),
            "distribution_loss": float(distribution_loss.mean().item()),
        }

    def transition_update(self, buffer):
        """
        Only update transition model (flow model). Corresponds to one iteration of model learning.
        
        Args:
            buffer (common.buffer.Buffer): Replay buffer.
        
        Returns:
            dict: Dictionary of training statistics.
        """
        obs, action, reward, next_s_samples, r_samples, task = buffer.sample()
        obs = obs.float()
        action = action.float()
        next_s_samples = next_s_samples.float()
        task = torch.tensor([0])

        # Compute targets
        with torch.no_grad():
            next_z = self.model.encode(obs[1:], task)
            td_targets = self._td_target(next_z, reward, task)
            
            # multiple next_z samples
            next_z_samples = torch.empty(self.cfg.horizon, self.cfg.batch_size, next_s_samples.shape[2], self.cfg.latent_dim, device=self.device)
            for i in range(next_s_samples.shape[2]):
                next_z_samples[:, :, i, :] = self.model.encode(next_s_samples[:, :, i, :], task)

        # Prepare for update
        self.optim.zero_grad(set_to_none=True)
        self.model.train_transition()

        # Latent rollout
        zs = torch.empty(self.cfg.horizon+1, self.cfg.batch_size, self.cfg.latent_dim, device=self.device)
        z = self.model.encode(obs[0], task)
        zs[0] = z
        consistency_loss = 0
        distribution_loss = 0
        first_distribution_loss = None
        for t in range(self.cfg.horizon):
            z = self.model.next(z, action[t], task)
            consistency_loss += F.mse_loss(z, next_z[t]) * self.cfg.rho**t
            repeated_z = z.unsqueeze(1).repeat(1, next_z_samples.shape[2], 1)
            distribution_loss += F.kl_div(
                F.log_softmax(repeated_z, dim=2),
                F.softmax(next_z_samples[t], dim=2),
                reduction='batchmean'
            ) * self.cfg.rho**t
            if first_distribution_loss is None:
                first_distribution_loss = distribution_loss
            zs[t+1] = z

        # Compute losses
        consistency_loss *= (1/self.cfg.horizon)
        distribution_loss *= (1/self.cfg.horizon)
        total_loss = consistency_loss

        # Update model
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
        self.optim.step()

        # Return training statistics
        self.model.eval()
        return {
            "train/consistency_loss": float(consistency_loss.mean().item()),
            "train/total_loss": float(total_loss.mean().item()),
            "train/grad_norm": float(grad_norm),
            "train/first_distribution_loss": float(first_distribution_loss.mean().item()),
            "train/distribution_loss": float(distribution_loss.mean().item()),
        }


    def transition_eval(self, buffer):
        """
        Only eval transition model (flow model). Corresponds to one iteration of model learning.
        
        Args:
            buffer (common.buffer.Buffer): Replay buffer.
        
        Returns:
            dict: Dictionary of training statistics.
        """
        obs, action, reward, next_s_samples, r_samples, task = buffer.sample()
        obs = obs.float()
        action = action.float()
        next_s_samples = next_s_samples.float()
        self.model.eval()

        # Compute targets
        with torch.no_grad():
            next_z = self.model.encode(obs[1:], task)
            td_targets = self._td_target(next_z, reward, task)
        
            # multiple next_z samples
            next_z_samples = torch.empty(self.cfg.horizon, self.cfg.batch_size, next_s_samples.shape[2], self.cfg.latent_dim, device=self.device)
            for i in range(next_s_samples.shape[2]):
                next_z_samples[:, :, i, :] = self.model.encode(next_s_samples[:, :, i, :], task)

        # Latent rollout
        zs = torch.empty(self.cfg.horizon+1, self.cfg.batch_size, self.cfg.latent_dim, device=self.device)
        z = self.model.encode(obs[0], task)
        zs[0] = z
        consistency_loss = 0
        distribution_loss = 0
        first_distribution_loss = None
        for t in range(self.cfg.horizon):
            z = self.model.next(z, action[t], task)
            consistency_loss += F.mse_loss(z, next_z[t]) * self.cfg.rho**t
            repeated_z = z.unsqueeze(1).repeat(1, next_z_samples.shape[2], 1)
            distribution_loss += F.kl_div(
                F.log_softmax(repeated_z, dim=2),
                F.softmax(next_z_samples[t], dim=2),
                reduction='batchmean'
            ) * self.cfg.rho**t
            if first_distribution_loss is None:
                first_distribution_loss = distribution_loss
            zs[t+1] = z
            
        
        # Compute losses
        consistency_loss *= (1/self.cfg.horizon)
        distribution_loss *= (1/self.cfg.horizon)
        total_loss = consistency_loss

        # Return training statistics
        return {
            "eval/consistency_loss": float(consistency_loss.mean().item()),
            "eval/total_loss": float(total_loss.mean().item()),
            "eval/first_distribution_loss": float(first_distribution_loss.mean().item()),
            "eval/distribution_loss": float(distribution_loss.mean().item()),
        }


class TDMPC2_Flow:
    """
    TD-MPC2 agent. Implements training + inference.
    Can be used for both single-task and multi-task experiments,
    and supports both state and pixel observations.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device('cuda')
        self.model = WorldModel_Flow(cfg, self.device).to(self.device)
        if cfg.pretrained_path:
            self.load(cfg.pretrained_path)
            print(f'loaded pretrained model from', colored(cfg.pretrained_path, 'yellow', attrs=['bold']))
        else:
            print(colored('train from scratch', 'yellow', attrs=['bold']))
        # wd list
        dynamic_params_1, dynamic_params_2 = configure_weight_decay(self.model._dynamics, self.cfg.weight_decay)
        self.optim = torch.optim.Adam([
            {'params': self.model._encoder.parameters(), 'lr': self.cfg.lr*self.cfg.enc_lr_scale},
            dynamic_params_1, dynamic_params_2,
            {'params': self.model._reward.parameters()},
            {'params': self.model._Qs.parameters()},
            {'params': self.model._task_emb.parameters() if self.cfg.multitask else []}
        ], lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        self.pi_optim = torch.optim.Adam(self.model._pi.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay, eps=1e-5)
        self.model.eval()
        self.scale = RunningScale(cfg)
        self.cfg.iterations += 2*int(cfg.action_dim >= 20) # Heuristic for large action spaces
        self.discount = torch.tensor(
            [self._get_discount(ep_len) for ep_len in cfg.episode_lengths], device='cuda'
        ) if self.cfg.multitask else self._get_discount(cfg.episode_length)

    def _get_discount(self, episode_length):
        """
        Returns discount factor for a given episode length.
        Simple heuristic that scales discount linearly with episode length.
        Default values should work well for most tasks, but can be changed as needed.

        Args:
            episode_length (int): Length of the episode. Assumes episodes are of fixed length.

        Returns:
            float: Discount factor for the task.
        """
        frac = episode_length/self.cfg.discount_denom
        return min(max((frac-1)/(frac), self.cfg.discount_min), self.cfg.discount_max)

    def save(self, fp):
        """
        Save state dict of the agent to filepath.
        
        Args:
            fp (str): Filepath to save state dict to.
        """
        torch.save({"model": self.model.state_dict()}, fp)

    def load(self, fp):
        """
        Load a saved state dict from filepath (or dictionary) into current agent.
        
        Args:
            fp (str or dict): Filepath or state dict to load.
        """
        state_dict = fp if isinstance(fp, dict) else torch.load(fp)
        self.model.load_state_dict(state_dict["model"])

    @torch.no_grad()
    def act(self, obs, t0=False, eval_mode=False, task=None):
        """
        Select an action by planning in the latent space of the world model.
        
        Args:
            obs (torch.Tensor): Observation from the environment.
            t0 (bool): Whether this is the first observation in the episode.
            eval_mode (bool): Whether to use the mean of the action distribution.
            task (int): Task index (only used for multi-task experiments).
        
        Returns:
            torch.Tensor: Action to take in the environment.
        """
        obs = obs.to(self.device, non_blocking=True).unsqueeze(0)
        if self.cfg.mpc:
            a = self.plan(obs, t0=t0, eval_mode=eval_mode, task=task)
        else:
            z = self.model.encode(obs, task)
            a = self.model.pi(z, task)[int(not eval_mode)][0]
        return a.cpu()

    @torch.no_grad()
    def _estimate_value(self, obs, actions, task):
        """Estimate value of a trajectory starting at latent state z and executing given actions."""
        G, discount = 0, 1
        for t in range(self.cfg.horizon):
            z = self.model.encode(obs, task)
            reward = math.two_hot_inv(self.model.reward(z, actions[t], task), self.cfg)
            obs = self.model.next(obs, actions[t], t_step=self.cfg.consistency_t_step)
            G += discount * reward
            discount *= self.discount[torch.tensor(task)] if self.cfg.multitask else self.discount
        z = self.model.encode(obs, task)
        return G + discount * self.model.Q(z, self.model.pi(z, task)[1], task, return_type='avg')

    @torch.no_grad()
    def plan(self, obs, t0=False, eval_mode=False, task=None):
        """
        Plan a sequence of actions using the learned world model.
        
        Args:
            z (torch.Tensor): Latent state from which to plan.
            t0 (bool): Whether this is the first observation in the episode.
            eval_mode (bool): Whether to use the mean of the action distribution.
            task (Torch.Tensor): Task index (only used for multi-task experiments).

        Returns:
            torch.Tensor: Action to take in the environment.
        """        
        # Sample policy trajectories
        if self.cfg.num_pi_trajs > 0:
            pi_actions = torch.empty(self.cfg.horizon, self.cfg.num_pi_trajs, self.cfg.action_dim, device=self.device)
            _obs = obs.repeat(self.cfg.num_pi_trajs, 1)
            for t in range(self.cfg.horizon-1):
                _z = self.model.encode(_obs, task)
                pi_actions[t] = self.model.pi(_z, task)[1]
                _obs = self.model.next(_obs, pi_actions[t], t_step=self.cfg.consistency_t_step)
            _z = self.model.encode(_obs, task)
            pi_actions[-1] = self.model.pi(_z, task)[1]

        # Initialize state and parameters
        obs = obs.repeat(self.cfg.num_samples, 1)
        mean = torch.zeros(self.cfg.horizon, self.cfg.action_dim, device=self.device)
        std = self.cfg.max_std*torch.ones(self.cfg.horizon, self.cfg.action_dim, device=self.device)
        if not t0:
            mean[:-1] = self._prev_mean[1:]
        actions = torch.empty(self.cfg.horizon, self.cfg.num_samples, self.cfg.action_dim, device=self.device)
        if self.cfg.num_pi_trajs > 0:
            actions[:, :self.cfg.num_pi_trajs] = pi_actions
    
        # Iterate MPPI
        for _ in range(self.cfg.iterations):

            # Sample actions
            actions[:, self.cfg.num_pi_trajs:] = (mean.unsqueeze(1) + std.unsqueeze(1) * \
                torch.randn(self.cfg.horizon, self.cfg.num_samples-self.cfg.num_pi_trajs, self.cfg.action_dim, device=std.device)) \
                .clamp(-1, 1)
            if self.cfg.multitask:
                actions = actions * self.model._action_masks[task]

            # Compute elite actions
            value = self._estimate_value(obs, actions, task).nan_to_num_(0)
            elite_idxs = torch.topk(value.squeeze(1), self.cfg.num_elites, dim=0).indices
            elite_value, elite_actions = value[elite_idxs], actions[:, elite_idxs]

            # Update parameters
            max_value = elite_value.max(0)[0]
            score = torch.exp(self.cfg.temperature*(elite_value - max_value))
            score /= score.sum(0)
            mean = torch.sum(score.unsqueeze(0) * elite_actions, dim=1) / (score.sum(0) + 1e-9)
            std = torch.sqrt(torch.sum(score.unsqueeze(0) * (elite_actions - mean.unsqueeze(1)) ** 2, dim=1) / (score.sum(0) + 1e-9)) \
                .clamp_(self.cfg.min_std, self.cfg.max_std)
            if self.cfg.multitask:
                mean = mean * self.model._action_masks[task]
                std = std * self.model._action_masks[task]

        # Select action
        score = score.squeeze(1).cpu().numpy()
        actions = elite_actions[:, np.random.choice(np.arange(score.shape[0]), p=score)]
        self._prev_mean = mean
        a, std = actions[0], std[0]
        if not eval_mode:
            a += std * torch.randn(self.cfg.action_dim, device=std.device)
        return a.clamp_(-1, 1)
        
    def update_pi(self, zs, task):
        """
        Update policy using a sequence of latent states.
        
        Args:
            zs (torch.Tensor): Sequence of latent states.
            task (torch.Tensor): Task index (only used for multi-task experiments).

        Returns:
            float: Loss of the policy update.
        """
        self.pi_optim.zero_grad(set_to_none=True)
        self.model.track_q_grad(False)
        _, pis, log_pis, _ = self.model.pi(zs, task)
        qs = self.model.Q(zs, pis, task, return_type='avg')
        self.scale.update(qs[0])
        qs = self.scale(qs)

        # Loss is a weighted sum of Q-values
        rho = torch.pow(self.cfg.rho, torch.arange(len(qs), device=self.device))
        pi_loss = ((self.cfg.entropy_coef * log_pis - qs).mean(dim=(1,2)) * rho).mean()
        pi_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model._pi.parameters(), self.cfg.grad_clip_norm)
        self.pi_optim.step()
        self.model.track_q_grad(True)

        return pi_loss.item()

    @torch.no_grad()
    def _td_target(self, next_z, reward, task):
        """
        Compute the TD-target from a reward and the observation at the following time step.
        
        Args:
            next_z (torch.Tensor): Latent state at the following time step.
            reward (torch.Tensor): Reward at the current time step.
            task (torch.Tensor): Task index (only used for multi-task experiments).
        
        Returns:
            torch.Tensor: TD-target.
        """
        pi = self.model.pi(next_z, task)[1]
        discount = self.discount[task].unsqueeze(-1) if self.cfg.multitask else self.discount
        return reward + discount * self.model.Q(next_z, pi, task, return_type='min', target=True)

    def update(self, buffer):
        """
        Main update function. Corresponds to one iteration of model learning.
        
        Args:
            buffer (common.buffer.Buffer): Replay buffer.
        
        Returns:
            dict: Dictionary of training statistics.
        """
        # TODO: task??
        obs, action, reward, next_s_samples, r_samples, task = buffer.sample()
        obs = obs.float()
        action = action.float()
        next_s_samples = next_s_samples.float()
    
        # Compute targets only (no use of next_z)
        with torch.no_grad():
            next_z = self.model.encode(obs[1:], task)
            td_targets = self._td_target(next_z, reward, task)

        # Prepare for update
        self.optim.zero_grad(set_to_none=True)
        self.model.train()

        # zs = torch.empty(self.cfg.horizon+1, self.cfg.batch_size, self.cfg.latent_dim, device=self.device)
        states = torch.empty(self.cfg.horizon+1, self.cfg.batch_size, self.cfg.state_dim, device=self.device)
        multiple_states = torch.empty(self.cfg.horizon, self.cfg.batch_size, next_s_samples.shape[2], self.cfg.state_dim, device=self.device)
        # z = self.model.encode(obs[0], task)
        # zs[0] = z
        s = obs[0]
        states[0] = s
        consistency_loss = 0
        flow_matching_loss = 0
        distribution_loss = 0
        first_distribution_loss = None
        for t in range(self.cfg.horizon):
            s0 = s
            s = self.model.next(s0, action[t], t_step=self.cfg.consistency_t_step)  # raw
            for i in range(next_s_samples.shape[2]):
                multiple_states[t, :, i, :] = self.model.next(s0, action[t], t_step=self.cfg.consistency_t_step)
            consistency_loss += F.mse_loss(s, obs[t]) * self.cfg.rho**t
            flow_matching_loss += self.model._dynamics.flow_matching_loss(x0=s0.detach(), x1=obs[t], condition=action[t]) * self.cfg.rho**t
            distribution_loss += F.kl_div(
                F.log_softmax(multiple_states[t], dim=2),
                F.softmax(next_s_samples[t], dim=2),
                reduction='batchmean'
            ) * self.cfg.rho**t
            if first_distribution_loss is None:
                first_distribution_loss = distribution_loss
            states[t+1] = s

        # Predictions
        zs = self.model.encode(states, task)
        _zs = zs[:-1]
        qs = self.model.Q(_zs, action, task, return_type='all')
        reward_preds = self.model.reward(_zs, action, task)
        
        # Compute losses
        reward_loss, value_loss = 0, 0
        for t in range(self.cfg.horizon):
            reward_loss += math.soft_ce(reward_preds[t], reward[t], self.cfg).mean() * self.cfg.rho**t
            for q in range(self.cfg.num_q):
                value_loss += math.soft_ce(qs[q][t], td_targets[t], self.cfg).mean() * self.cfg.rho**t
        consistency_loss *= (1/self.cfg.horizon)
        reward_loss *= (1/self.cfg.horizon)
        value_loss *= (1/(self.cfg.horizon * self.cfg.num_q))
        flow_matching_loss *= (1/self.cfg.horizon)
        distribution_loss *= (1/self.cfg.horizon)
        total_loss = (
            self.cfg.consistency_coef * consistency_loss +
            self.cfg.reward_coef * reward_loss +
            self.cfg.value_coef * value_loss +
            self.cfg.flow_coef * flow_matching_loss
        )

        # Update model
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
        self.optim.step()

        # Update policy
        pi_loss = self.update_pi(zs.detach(), task)

        # Update target Q-functions
        self.model.soft_update_target_Q()

        # Return training statistics
        self.model.eval()
        return {
            "consistency_loss": float(consistency_loss.mean().item()),
            "reward_loss": float(reward_loss.mean().item()),
            "value_loss": float(value_loss.mean().item()),
            "pi_loss": pi_loss,
            "total_loss": float(total_loss.mean().item()),
            "flow_matching_loss": float(flow_matching_loss.mean().item()),
            "grad_norm": float(grad_norm),
            "pi_scale": float(self.scale.value),
            "distribution_loss": float(distribution_loss.mean().item()),
            "first_distribution_loss": float(first_distribution_loss.mean().item()),
        }

    def transition_update(self, buffer):
        """
        Only update transition model (flow model). Corresponds to one iteration of model learning.
        
        Args:
            buffer (common.buffer.Buffer): Replay buffer.
        
        Returns:
            dict: Dictionary of training statistics.
        """
        obs, action, reward, next_s_samples, r_samples, task = buffer.sample()
        obs = obs.float()
        action = action.float()
        next_s_samples = next_s_samples.float()

        # Prepare for update
        self.optim.zero_grad(set_to_none=True)
        self.model.train_transition()

        # zs = torch.empty(self.cfg.horizon+1, self.cfg.batch_size, self.cfg.latent_dim, device=self.device)
        states = torch.empty(self.cfg.horizon+1, self.cfg.batch_size, self.cfg.state_dim, device=self.device)
        multiple_states = torch.empty(self.cfg.horizon, self.cfg.batch_size, next_s_samples.shape[2], self.cfg.state_dim, device=self.device)
        # z = self.model.encode(obs[0], task)
        # zs[0] = z
        s = obs[0]
        states[0] = s
        consistency_loss = 0
        flow_matching_loss = 0
        distribution_loss = 0
        first_distribution_loss = None
        for t in range(self.cfg.horizon):
            s0 = s
            s = self.model.next(s0, action[t], t_step=self.cfg.consistency_t_step)  # raw
            for i in range(next_s_samples.shape[2]):
                multiple_states[t, :, i, :] = self.model.next(s0, action[t], t_step=self.cfg.consistency_t_step)
            consistency_loss += F.mse_loss(s, obs[t]) * self.cfg.rho**t
            # https://github.com/opendilab/GenerativeRL/blob/3e1172ae0cbe18f311d40926d1e485b135a8e92c/grl/generative_models/model_functions/velocity_function.py#L220
            flow_matching_loss += self.model._dynamics.flow_matching_loss(x0=s0.detach(), x1=obs[t], condition=action[t]) * self.cfg.rho**t
            distribution_loss += F.kl_div(
                F.log_softmax(multiple_states[t], dim=2),
                F.softmax(next_s_samples[t], dim=2),
                reduction='batchmean'
            ) * self.cfg.rho**t
            if first_distribution_loss is None:
                first_distribution_loss = distribution_loss
            states[t+1] = s
        
        consistency_loss *= (1/self.cfg.horizon)
        flow_matching_loss *= (1/self.cfg.horizon)
        distribution_loss *= (1/self.cfg.horizon)
        total_loss = (
            self.cfg.consistency_coef * consistency_loss +
            self.cfg.flow_coef * flow_matching_loss
        )

        # Update model
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
        self.optim.step()

        # Return training statistics
        self.model.eval()
        return {
            "train/consistency_loss": float(consistency_loss.mean().item()),
            "train/total_loss": float(total_loss.mean().item()),
            "train/flow_matching_loss": float(flow_matching_loss.mean().item()),
            "train/grad_norm": float(grad_norm),
            "train/distribution_loss": float(distribution_loss.mean().item()),
            "train/first_distribution_loss": float(first_distribution_loss.mean().item()),
        }


    def transition_eval(self, buffer):
        """
        Only eval transition model (flow model). Corresponds to one iteration of model learning.
        
        Args:
            buffer (common.buffer.Buffer): Replay buffer.
        
        Returns:
            dict: Dictionary of training statistics.
        """
        obs, action, reward, next_s_samples, r_samples, task = buffer.sample()
        obs = obs.float()
        action = action.float()
        next_s_samples = next_s_samples.float()
        self.model.eval()

        # zs = torch.empty(self.cfg.horizon+1, self.cfg.batch_size, self.cfg.latent_dim, device=self.device)
        states = torch.empty(self.cfg.horizon+1, self.cfg.batch_size, self.cfg.state_dim, device=self.device)
        multiple_states = torch.empty(self.cfg.horizon, self.cfg.batch_size, next_s_samples.shape[2], self.cfg.state_dim, device=self.device)
        # z = self.model.encode(obs[0], task)
        # zs[0] = z
        s = obs[0]
        states[0] = s
        consistency_loss = 0
        flow_matching_loss = 0
        distribution_loss = 0
        first_distribution_loss = None
        for t in range(self.cfg.horizon):
            s0 = s
            s = self.model.next(s0, action[t])  # raw
            for i in range(next_s_samples.shape[2]):
                multiple_states[t, :, i, :] = self.model.next(s0, action[t], t_step=self.cfg.consistency_t_step)
            consistency_loss += F.mse_loss(s, obs[t]) * self.cfg.rho**t
            # https://github.com/opendilab/GenerativeRL/blob/3e1172ae0cbe18f311d40926d1e485b135a8e92c/grl/generative_models/model_functions/velocity_function.py#L220
            flow_matching_loss += self.model._dynamics.flow_matching_loss(x0=s0.detach(), x1=obs[t], condition=action[t]) * self.cfg.rho**t
            distribution_loss += F.kl_div(
                F.log_softmax(multiple_states[t], dim=2),
                F.softmax(next_s_samples[t], dim=2),
                reduction='batchmean'
            ) * self.cfg.rho**t
            if first_distribution_loss is None:
                first_distribution_loss = distribution_loss
            states[t+1] = s
        
        consistency_loss *= (1/self.cfg.horizon)
        flow_matching_loss *= (1/self.cfg.horizon)
        distribution_loss *= (1/self.cfg.horizon)
        total_loss = (
            self.cfg.consistency_coef * consistency_loss +
            self.cfg.flow_coef * flow_matching_loss
        )

        # Return training statistics
        return {
            "eval/consistency_loss": float(consistency_loss.mean().item()),
            "eval/total_loss": float(total_loss.mean().item()),
            "eval/flow_matching_loss": float(flow_matching_loss.mean().item()),
            "eval/distribution_loss": float(distribution_loss.mean().item()),
            "eval/first_distribution_loss": float(first_distribution_loss.mean().item()),
        }



class TDMPC2_Flow_MultiGPU:
    """
    TD-MPC2 agent. Implements training + inference.
    Can be used for both single-task and multi-task experiments,
    and supports both state and pixel observations.
    """

    def __init__(self, cfg, accelerator):
        self.cfg = cfg
        self.device = accelerator.device
        self.model = WorldModel_Flow(cfg, self.device).to(self.device)
        if cfg.pretrained_path:
            self.load(cfg.pretrained_path)
            print(f'loaded pretrained model from', colored(cfg.cfg.pretrained_path, 'yellow', attrs=['bold']))
        else:
            print(colored('train from scratch', 'yellow', attrs=['bold']))
        self.optim = torch.optim.Adam([
            {'params': self.model._encoder.parameters(), 'lr': self.cfg.lr*self.cfg.enc_lr_scale},
            {'params': self.model._dynamics.parameters()},
            {'params': self.model._reward.parameters()},
            {'params': self.model._Qs.parameters()},
            {'params': self.model._task_emb.parameters() if self.cfg.multitask else []}
        ], lr=self.cfg.lr)
        self.pi_optim = torch.optim.Adam(self.model._pi.parameters(), lr=self.cfg.lr, eps=1e-5)
        self.model.eval()
        self.scale = RunningScale(cfg)
        self.cfg.iterations += 2*int(cfg.action_dim >= 20) # Heuristic for large action spaces
        self.discount = torch.tensor(
            [self._get_discount(ep_len) for ep_len in cfg.episode_lengths], device='cuda'
        ) if self.cfg.multitask else self._get_discount(cfg.episode_length)

        # self.model._encoder = accelerator.prepare(self.model._encoder)
        for key, module in self.model._encoder.items():
            self.model._encoder[key] = accelerator.prepare(module)

        self.model._dynamics.model = accelerator.prepare(self.model._dynamics.model)
        self.model._reward = accelerator.prepare(self.model._reward)
        self.model._Qs = accelerator.prepare(self.model._Qs)
        self.model._task_emb = accelerator.prepare(self.model._task_emb)
        self.model._pi = accelerator.prepare(self.model._pi)

        self.optim = accelerator.prepare(self.optim)
        self.pi_optim = accelerator.prepare(self.pi_optim)

    def _get_discount(self, episode_length):
        """
        Returns discount factor for a given episode length.
        Simple heuristic that scales discount linearly with episode length.
        Default values should work well for most tasks, but can be changed as needed.

        Args:
            episode_length (int): Length of the episode. Assumes episodes are of fixed length.

        Returns:
            float: Discount factor for the task.
        """
        frac = episode_length/self.cfg.discount_denom
        return min(max((frac-1)/(frac), self.cfg.discount_min), self.cfg.discount_max)

    def save(self, fp):
        """
        Save state dict of the agent to filepath.
        
        Args:
            fp (str): Filepath to save state dict to.
        """
        torch.save({"model": self.model.state_dict()}, fp)

    def load(self, fp):
        """
        Load a saved state dict from filepath (or dictionary) into current agent.
        
        Args:
            fp (str or dict): Filepath or state dict to load.
        """
        state_dict = fp if isinstance(fp, dict) else torch.load(fp)
        self.model.load_state_dict(state_dict["model"])

    @torch.no_grad()
    def act(self, obs, t0=False, eval_mode=False, task=None):
        """
        Select an action by planning in the latent space of the world model.
        
        Args:
            obs (torch.Tensor): Observation from the environment.
            t0 (bool): Whether this is the first observation in the episode.
            eval_mode (bool): Whether to use the mean of the action distribution.
            task (int): Task index (only used for multi-task experiments).
        
        Returns:
            torch.Tensor: Action to take in the environment.
        """
        obs = obs.to(self.device, non_blocking=True).unsqueeze(0)
        if task is not None:
            task = torch.tensor([task], device=self.device)
        z = self.model.encode(obs, task)
        if self.cfg.mpc:
            a = self.plan(z, t0=t0, eval_mode=eval_mode, task=task)
        else:
            a = self.model.pi(z, task)[int(not eval_mode)][0]
        return a.cpu()

    @torch.no_grad()
    def _estimate_value(self, z, actions, task):
        """Estimate value of a trajectory starting at latent state z and executing given actions."""
        G, discount = 0, 1
        for t in range(self.cfg.horizon):
            reward = math.two_hot_inv(self.model.reward(z, actions[t], task), self.cfg)
            z = self.model.next(z, actions[t], task)
            G += discount * reward
            discount *= self.discount[torch.tensor(task)] if self.cfg.multitask else self.discount
        return G + discount * self.model.Q(z, self.model.pi(z, task)[1], task, return_type='avg')

    @torch.no_grad()
    def plan(self, z, t0=False, eval_mode=False, task=None):
        """
        Plan a sequence of actions using the learned world model.
        
        Args:
            z (torch.Tensor): Latent state from which to plan.
            t0 (bool): Whether this is the first observation in the episode.
            eval_mode (bool): Whether to use the mean of the action distribution.
            task (Torch.Tensor): Task index (only used for multi-task experiments).

        Returns:
            torch.Tensor: Action to take in the environment.
        """        
        # Sample policy trajectories
        if self.cfg.num_pi_trajs > 0:
            pi_actions = torch.empty(self.cfg.horizon, self.cfg.num_pi_trajs, self.cfg.action_dim, device=self.device)
            _z = z.repeat(self.cfg.num_pi_trajs, 1)
            for t in range(self.cfg.horizon-1):
                pi_actions[t] = self.model.pi(_z, task)[1]
                _z = self.model.next(_z, pi_actions[t], task)
            pi_actions[-1] = self.model.pi(_z, task)[1]

        # Initialize state and parameters
        z = z.repeat(self.cfg.num_samples, 1)
        mean = torch.zeros(self.cfg.horizon, self.cfg.action_dim, device=self.device)
        std = self.cfg.max_std*torch.ones(self.cfg.horizon, self.cfg.action_dim, device=self.device)
        if not t0:
            mean[:-1] = self._prev_mean[1:]
        actions = torch.empty(self.cfg.horizon, self.cfg.num_samples, self.cfg.action_dim, device=self.device)
        if self.cfg.num_pi_trajs > 0:
            actions[:, :self.cfg.num_pi_trajs] = pi_actions
    
        # Iterate MPPI
        for _ in range(self.cfg.iterations):

            # Sample actions
            actions[:, self.cfg.num_pi_trajs:] = (mean.unsqueeze(1) + std.unsqueeze(1) * \
                torch.randn(self.cfg.horizon, self.cfg.num_samples-self.cfg.num_pi_trajs, self.cfg.action_dim, device=std.device)) \
                .clamp(-1, 1)
            if self.cfg.multitask:
                actions = actions * self.model._action_masks[task]

            # Compute elite actions
            value = self._estimate_value(z, actions, task).nan_to_num_(0)
            elite_idxs = torch.topk(value.squeeze(1), self.cfg.num_elites, dim=0).indices
            elite_value, elite_actions = value[elite_idxs], actions[:, elite_idxs]

            # Update parameters
            max_value = elite_value.max(0)[0]
            score = torch.exp(self.cfg.temperature*(elite_value - max_value))
            score /= score.sum(0)
            mean = torch.sum(score.unsqueeze(0) * elite_actions, dim=1) / (score.sum(0) + 1e-9)
            std = torch.sqrt(torch.sum(score.unsqueeze(0) * (elite_actions - mean.unsqueeze(1)) ** 2, dim=1) / (score.sum(0) + 1e-9)) \
                .clamp_(self.cfg.min_std, self.cfg.max_std)
            if self.cfg.multitask:
                mean = mean * self.model._action_masks[task]
                std = std * self.model._action_masks[task]

        # Select action
        score = score.squeeze(1).cpu().numpy()
        actions = elite_actions[:, np.random.choice(np.arange(score.shape[0]), p=score)]
        self._prev_mean = mean
        a, std = actions[0], std[0]
        if not eval_mode:
            a += std * torch.randn(self.cfg.action_dim, device=std.device)
        return a.clamp_(-1, 1)
        
    def update_pi(self, zs, task, accelerator):
        """
        Update policy using a sequence of latent states.
        
        Args:
            zs (torch.Tensor): Sequence of latent states.
            task (torch.Tensor): Task index (only used for multi-task experiments).

        Returns:
            float: Loss of the policy update.
        """
        self.pi_optim.zero_grad(set_to_none=True)
        self.model.track_q_grad(False)
        _, pis, log_pis, _ = self.model.pi(zs, task)
        qs = self.model.Q(zs, pis, task, return_type='avg')
        self.scale.update(qs[0])
        qs = self.scale(qs)

        # Loss is a weighted sum of Q-values
        rho = torch.pow(self.cfg.rho, torch.arange(len(qs), device=self.device))
        pi_loss = ((self.cfg.entropy_coef * log_pis - qs).mean(dim=(1,2)) * rho).mean()
        accelerator.backward(pi_loss)
        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(self.model._pi.parameters(), self.cfg.grad_clip_norm)
        self.pi_optim.step()
        self.model.track_q_grad(True)

        return pi_loss.item()

    @torch.no_grad()
    def _td_target(self, next_z, reward, task):
        """
        Compute the TD-target from a reward and the observation at the following time step.
        
        Args:
            next_z (torch.Tensor): Latent state at the following time step.
            reward (torch.Tensor): Reward at the current time step.
            task (torch.Tensor): Task index (only used for multi-task experiments).
        
        Returns:
            torch.Tensor: TD-target.
        """
        pi = self.model.pi(next_z, task)[1]
        discount = self.discount[task].unsqueeze(-1) if self.cfg.multitask else self.discount
        return reward + discount * self.model.Q(next_z, pi, task, return_type='min', target=True)

    def update(self, buffer, accelerator):
        """
        Main update function. Corresponds to one iteration of model learning.
        
        Args:
            buffer (common.buffer.Buffer): Replay buffer.
        
        Returns:
            dict: Dictionary of training statistics.
        """
        with accelerator.autocast():
            obs, action, reward, task = buffer.sample()
        
            # Compute targets
            with torch.no_grad():
                next_z = self.model.encode(obs[1:], task)
                td_targets = self._td_target(next_z, reward, task)

            # Prepare for update
            self.optim.zero_grad(set_to_none=True)
            self.model.train()

            # Latent rollout
            zs = torch.empty(self.cfg.horizon+1, self.cfg.batch_size, self.cfg.latent_dim, device=self.device)
            z = self.model.encode(obs[0], task)
            zs[0] = z
            consistency_loss = 0
            flow_matching_loss = 0
            for t in range(self.cfg.horizon):
                z0 = z
                z = self.model.next(z, action[t], task)
                consistency_loss += F.mse_loss(z, next_z[t]) * self.cfg.rho**t
                if self.model.cfg.flow_model == 'unet':
                    condition_t = self.model.task_emb(action[t], task)
                else:
                    task_emb = self.model._task_emb(task.long())
                    condition_t = TensorDict({'action': action[t], 'background': task_emb})
                flow_matching_loss += self.model._dynamics.flow_matching_loss(x0=z0.detach(), x1=next_z[t], condition=condition_t) * self.cfg.rho**t
                zs[t+1] = z

            # Predictions
            _zs = zs[:-1]
            qs = self.model.Q(_zs, action, task, return_type='all')
            reward_preds = self.model.reward(_zs, action, task)
            
            # Compute losses
            reward_loss, value_loss = 0, 0
            for t in range(self.cfg.horizon):
                reward_loss += math.soft_ce(reward_preds[t], reward[t], self.cfg).mean() * self.cfg.rho**t
                for q in range(self.cfg.num_q):
                    value_loss += math.soft_ce(qs[q][t], td_targets[t], self.cfg).mean() * self.cfg.rho**t
            consistency_loss *= (1/self.cfg.horizon)
            reward_loss *= (1/self.cfg.horizon)
            value_loss *= (1/(self.cfg.horizon * self.cfg.num_q))
            flow_matching_loss *= (1/self.cfg.horizon)
            total_loss = (
                self.cfg.consistency_coef * consistency_loss +
                self.cfg.reward_coef * reward_loss +
                self.cfg.value_coef * value_loss +
                1 * flow_matching_loss
            )

            # Update model
            accelerator.backward(total_loss)
            if accelerator.sync_gradients:
                grad_norm=accelerator.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
            self.optim.step()

            # Update policy
            pi_loss = self.update_pi(zs.detach(), task, accelerator)

            # Update target Q-functions
            self.model.soft_update_target_Q()

            # Return training statistics
            self.model.eval()

        return {
            "consistency_loss": float(consistency_loss.mean().item()),
            "reward_loss": float(reward_loss.mean().item()),
            "value_loss": float(value_loss.mean().item()),
            "pi_loss": pi_loss,
            "total_loss": float(total_loss.mean().item()),
            "flow_matching_loss": float(flow_matching_loss.mean().item()),
            "grad_norm": float(grad_norm),
            "pi_scale": float(self.scale.value),
        }

class TDMPC2_MultiGPU:
    """
    TD-MPC2 agent. Implements training + inference.
    Can be used for both single-task and multi-task experiments,
    and supports both state and pixel observations.
    """

    def __init__(self, cfg, accelerator):
        self.cfg = cfg
        self.device = accelerator.device
        self.model = WorldModel(cfg).to(self.device)
        if cfg.pretrained_path:
            self.load(cfg.pretrained_path)
            print(f'loaded pretrained model from', colored(cfg.pretrained_path, 'yellow', attrs=['bold']))
        else:
            print(colored('train from scratch', 'yellow', attrs=['bold']))
        self.optim = torch.optim.Adam([
            {'params': self.model._encoder.parameters(), 'lr': self.cfg.lr*self.cfg.enc_lr_scale},
            {'params': self.model._dynamics.parameters()},
            {'params': self.model._reward.parameters()},
            {'params': self.model._Qs.parameters()},
            {'params': self.model._task_emb.parameters() if self.cfg.multitask else []}
        ], lr=self.cfg.lr)
        self.pi_optim = torch.optim.Adam(self.model._pi.parameters(), lr=self.cfg.lr, eps=1e-5)
        self.model.eval()
        self.scale = RunningScale(cfg)
        self.cfg.iterations += 2*int(cfg.action_dim >= 20) # Heuristic for large action spaces
        self.discount = torch.tensor(
            [self._get_discount(ep_len) for ep_len in cfg.episode_lengths], device='cuda'
        ) if self.cfg.multitask else self._get_discount(cfg.episode_length)

        # self.model._encoder = accelerator.prepare(self.model._encoder)
        for key, module in self.model._encoder.items():
            self.model._encoder[key] = accelerator.prepare(module)

        self.model._dynamics = accelerator.prepare(self.model._dynamics)
        self.model._reward = accelerator.prepare(self.model._reward)
        self.model._Qs = accelerator.prepare(self.model._Qs)
        self.model._task_emb = accelerator.prepare(self.model._task_emb)
        self.model._pi = accelerator.prepare(self.model._pi)

        self.optim = accelerator.prepare(self.optim)
        self.pi_optim = accelerator.prepare(self.pi_optim)

    def _get_discount(self, episode_length):
        """
        Returns discount factor for a given episode length.
        Simple heuristic that scales discount linearly with episode length.
        Default values should work well for most tasks, but can be changed as needed.

        Args:
            episode_length (int): Length of the episode. Assumes episodes are of fixed length.

        Returns:
            float: Discount factor for the task.
        """
        frac = episode_length/self.cfg.discount_denom
        return min(max((frac-1)/(frac), self.cfg.discount_min), self.cfg.discount_max)

    def save(self, fp):
        """
        Save state dict of the agent to filepath.
        
        Args:
            fp (str): Filepath to save state dict to.
        """
        torch.save({"model": self.model.state_dict()}, fp)

    def load(self, fp):
        """
        Load a saved state dict from filepath (or dictionary) into current agent.
        
        Args:
            fp (str or dict): Filepath or state dict to load.
        """
        def load_weights(weight):
            # torch2.0版本以上支持torch.compile来跑模型，会快，但是compile还只支持linux系统
            # compile后的模型存权重的时候层的名字前面会加上'_orig_mod.'
            # 这段代码就是把这个删掉
            # 传入模型的路径（.pth），返回权重，直接使用model.load_state_dict(weight)就能读进去
            new_weight = weight.copy()
            keys_list = list(weight.keys())
            for key, orig_key in zip(keys_list, self.model.state_dict()):
                if not key == orig_key:
                    if 'orig_mod.module.' in key:
                        del_key = key.replace('_orig_mod.module.', '')
                        new_weight[del_key] = weight[key]
                        del new_weight[key]
            return new_weight
        state_dict = fp if isinstance(fp, dict) else torch.load(fp)
        new_weight = load_weights(state_dict["model"])
        self.model.load_state_dict(new_weight)

    @torch.no_grad()
    def act(self, obs, t0=False, eval_mode=False, task=None):
        """
        Select an action by planning in the latent space of the world model.
        
        Args:
            obs (torch.Tensor): Observation from the environment.
            t0 (bool): Whether this is the first observation in the episode.
            eval_mode (bool): Whether to use the mean of the action distribution.
            task (int): Task index (only used for multi-task experiments).
        
        Returns:
            torch.Tensor: Action to take in the environment.
        """
        obs = obs.to(self.device, non_blocking=True).unsqueeze(0)
        if task is not None:
            task = torch.tensor([task], device=self.device)
        z = self.model.encode(obs, task)
        if self.cfg.mpc:
            a = self.plan(z, t0=t0, eval_mode=eval_mode, task=task)
        else:
            a = self.model.pi(z, task)[int(not eval_mode)][0]
        return a.cpu()

    @torch.no_grad()
    def _estimate_value(self, z, actions, task):
        """Estimate value of a trajectory starting at latent state z and executing given actions."""
        G, discount = 0, 1
        for t in range(self.cfg.horizon):
            reward = math.two_hot_inv(self.model.reward(z, actions[t], task), self.cfg)
            z = self.model.next(z, actions[t], task)
            G += discount * reward
            discount *= self.discount[torch.tensor(task)] if self.cfg.multitask else self.discount
        return G + discount * self.model.Q(z, self.model.pi(z, task)[1], task, return_type='avg')

    @torch.no_grad()
    def plan(self, z, t0=False, eval_mode=False, task=None):
        """
        Plan a sequence of actions using the learned world model.
        
        Args:
            z (torch.Tensor): Latent state from which to plan.
            t0 (bool): Whether this is the first observation in the episode.
            eval_mode (bool): Whether to use the mean of the action distribution.
            task (Torch.Tensor): Task index (only used for multi-task experiments).

        Returns:
            torch.Tensor: Action to take in the environment.
        """        
        # Sample policy trajectories
        if self.cfg.num_pi_trajs > 0:
            pi_actions = torch.empty(self.cfg.horizon, self.cfg.num_pi_trajs, self.cfg.action_dim, device=self.device)
            _z = z.repeat(self.cfg.num_pi_trajs, 1)
            for t in range(self.cfg.horizon-1):
                pi_actions[t] = self.model.pi(_z, task)[1]
                _z = self.model.next(_z, pi_actions[t], task)
            pi_actions[-1] = self.model.pi(_z, task)[1]

        # Initialize state and parameters
        z = z.repeat(self.cfg.num_samples, 1)
        mean = torch.zeros(self.cfg.horizon, self.cfg.action_dim, device=self.device)
        std = self.cfg.max_std*torch.ones(self.cfg.horizon, self.cfg.action_dim, device=self.device)
        if not t0:
            mean[:-1] = self._prev_mean[1:]
        actions = torch.empty(self.cfg.horizon, self.cfg.num_samples, self.cfg.action_dim, device=self.device)
        if self.cfg.num_pi_trajs > 0:
            actions[:, :self.cfg.num_pi_trajs] = pi_actions
    
        # Iterate MPPI
        for _ in range(self.cfg.iterations):

            # Sample actions
            actions[:, self.cfg.num_pi_trajs:] = (mean.unsqueeze(1) + std.unsqueeze(1) * \
                torch.randn(self.cfg.horizon, self.cfg.num_samples-self.cfg.num_pi_trajs, self.cfg.action_dim, device=std.device)) \
                .clamp(-1, 1)
            if self.cfg.multitask:
                actions = actions * self.model._action_masks[task]

            # Compute elite actions
            value = self._estimate_value(z, actions, task).nan_to_num_(0)
            elite_idxs = torch.topk(value.squeeze(1), self.cfg.num_elites, dim=0).indices
            elite_value, elite_actions = value[elite_idxs], actions[:, elite_idxs]

            # Update parameters
            max_value = elite_value.max(0)[0]
            score = torch.exp(self.cfg.temperature*(elite_value - max_value))
            score /= score.sum(0)
            mean = torch.sum(score.unsqueeze(0) * elite_actions, dim=1) / (score.sum(0) + 1e-9)
            std = torch.sqrt(torch.sum(score.unsqueeze(0) * (elite_actions - mean.unsqueeze(1)) ** 2, dim=1) / (score.sum(0) + 1e-9)) \
                .clamp_(self.cfg.min_std, self.cfg.max_std)
            if self.cfg.multitask:
                mean = mean * self.model._action_masks[task]
                std = std * self.model._action_masks[task]

        # Select action
        score = score.squeeze(1).cpu().numpy()
        actions = elite_actions[:, np.random.choice(np.arange(score.shape[0]), p=score)]
        self._prev_mean = mean
        a, std = actions[0], std[0]
        if not eval_mode:
            a += std * torch.randn(self.cfg.action_dim, device=std.device)
        return a.clamp_(-1, 1)
        
    def update_pi(self, zs, task, accelerator):
        """
        Update policy using a sequence of latent states.
        
        Args:
            zs (torch.Tensor): Sequence of latent states.
            task (torch.Tensor): Task index (only used for multi-task experiments).

        Returns:
            float: Loss of the policy update.
        """
        self.pi_optim.zero_grad(set_to_none=True)
        self.model.track_q_grad(False)
        _, pis, log_pis, _ = self.model.pi(zs, task)
        qs = self.model.Q(zs, pis, task, return_type='avg')
        self.scale.update(qs[0])
        qs = self.scale(qs)

        # Loss is a weighted sum of Q-values
        rho = torch.pow(self.cfg.rho, torch.arange(len(qs), device=self.device))
        pi_loss = ((self.cfg.entropy_coef * log_pis - qs).mean(dim=(1,2)) * rho).mean()
        accelerator.backward(pi_loss)
        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(self.model._pi.parameters(), self.cfg.grad_clip_norm)
        self.pi_optim.step()
        self.model.track_q_grad(True)

        return pi_loss.item()

    @torch.no_grad()
    def _td_target(self, next_z, reward, task):
        """
        Compute the TD-target from a reward and the observation at the following time step.
        
        Args:
            next_z (torch.Tensor): Latent state at the following time step.
            reward (torch.Tensor): Reward at the current time step.
            task (torch.Tensor): Task index (only used for multi-task experiments).
        
        Returns:
            torch.Tensor: TD-target.
        """
        pi = self.model.pi(next_z, task)[1]
        discount = self.discount[task].unsqueeze(-1) if self.cfg.multitask else self.discount
        return reward + discount * self.model.Q(next_z, pi, task, return_type='min', target=True)

    def update(self, buffer, accelerator):
        """
        Main update function. Corresponds to one iteration of model learning.
        
        Args:
            buffer (common.buffer.Buffer): Replay buffer.
        
        Returns:
            dict: Dictionary of training statistics.
        """
        with accelerator.autocast():
            obs, action, reward, task = buffer.sample()
        
            # Compute targets
            with torch.no_grad():
                next_z = self.model.encode(obs[1:], task)
                td_targets = self._td_target(next_z, reward, task)

            # Prepare for update
            self.optim.zero_grad(set_to_none=True)
            self.model.train()

            # Latent rollout
            zs = torch.empty(self.cfg.horizon+1, self.cfg.batch_size, self.cfg.latent_dim, device=self.device)
            z = self.model.encode(obs[0], task)
            zs[0] = z
            consistency_loss = 0
            for t in range(self.cfg.horizon):
                z = self.model.next(z, action[t], task)
                consistency_loss += F.mse_loss(z, next_z[t]) * self.cfg.rho**t
                zs[t+1] = z

            # Predictions
            _zs = zs[:-1]
            qs = self.model.Q(_zs, action, task, return_type='all')
            reward_preds = self.model.reward(_zs, action, task)
            
            # Compute losses
            reward_loss, value_loss = 0, 0
            for t in range(self.cfg.horizon):
                reward_loss += math.soft_ce(reward_preds[t], reward[t], self.cfg).mean() * self.cfg.rho**t
                for q in range(self.cfg.num_q):
                    value_loss += math.soft_ce(qs[q][t], td_targets[t], self.cfg).mean() * self.cfg.rho**t
            consistency_loss *= (1/self.cfg.horizon)
            reward_loss *= (1/self.cfg.horizon)
            value_loss *= (1/(self.cfg.horizon * self.cfg.num_q))
            total_loss = (
                self.cfg.consistency_coef * consistency_loss +
                self.cfg.reward_coef * reward_loss +
                self.cfg.value_coef * value_loss
            )

            # Update model
            accelerator.backward(total_loss)
            if accelerator.sync_gradients:
                grad_norm=accelerator.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
            self.optim.step()

            # Update policy
            pi_loss = self.update_pi(zs.detach(), task, accelerator)

            # Update target Q-functions
            self.model.soft_update_target_Q()

            # Return training statistics
            self.model.eval()

        return {
            "consistency_loss": float(consistency_loss.mean().item()),
            "reward_loss": float(reward_loss.mean().item()),
            "value_loss": float(value_loss.mean().item()),
            "pi_loss": pi_loss,
            "total_loss": float(total_loss.mean().item()),
            "grad_norm": float(grad_norm),
            "pi_scale": float(self.scale.value),
        }
