import torch
import torch.nn.functional as F
import time
import einops

from common import math
from common.scale import RunningScale
from common.world_model_diffuser import WorldModelDiffuser
from tensordict import TensorDict


class TDMPC2Diffuser(torch.nn.Module):
    """
    TD-MPC2 agent. Implements training + inference.
    Can be used for both single-task and multi-task experiments,
    and supports both state and pixel observations.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device('cuda:0')
        self.model = WorldModelDiffuser(cfg).to(self.device)
        self.optim = torch.optim.Adam([
            {'params': self.model._encoder.parameters(), 'lr': self.cfg.lr*self.cfg.enc_lr_scale},
            # {'params': self.model._dynamics.parameters()},
            {'params': self.model._reward.parameters()},
            {'params': self.model._Qs.parameters()},
            {'params': self.model._task_emb.parameters() if self.cfg.multitask else []
             }
        ], lr=self.cfg.lr, capturable=True)
        # self.pi_optim = torch.optim.Adam(self.model._pi.parameters(), lr=self.cfg.lr, eps=1e-5, capturable=True)
        self.model.eval()
        self.scale = RunningScale(cfg)
        self.cfg.iterations += 2*int(cfg.action_dim >= 20) # Heuristic for large action spaces
        self.discount = torch.tensor(
            [self._get_discount(ep_len) for ep_len in cfg.episode_lengths], device='cuda:0'
        ) if self.cfg.multitask else self._get_discount(cfg.episode_length)
        self._prev_mean = torch.nn.Buffer(torch.zeros(self.cfg.horizon, self.cfg.action_dim, device=self.device))
        if cfg.compile:
            print('Compiling update function with torch.compile...')
            self._update = torch.compile(self._update, mode="reduce-overhead")

    @property
    def plan(self):
        _plan_val = getattr(self, "_plan_val", None)
        if _plan_val is not None:
            return _plan_val
        if self.cfg.compile:
            plan = torch.compile(self._plan, mode="reduce-overhead")
        else:
            plan = self._plan
        self._plan_val = plan
        return self._plan_val

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
        state_dict = state_dict["model"] if "model" in state_dict else state_dict
        try: # Checkpoints created AFTER Nov 10 update
            self.model.load_state_dict(state_dict)
        except: # Backwards compatibility
            def _get_submodule(state_dict, key):
                return {k.replace(f"_{key}.", ""): v for k, v in state_dict.items() if k.startswith(f"_{key}.")}
            for key in ["encoder", "dynamics", "reward", "pi"]:
                submodule_state_dict = _get_submodule(state_dict, key)
                getattr(self.model, f"_{key}").load_state_dict(submodule_state_dict)
            # Q-function requires special handling
            Qs_state_dict = _get_submodule(state_dict, "Qs")
            # TODO: figure out how to load Q-function state_dict from old checkpoints
            raise NotImplementedError("Backwards compatibility is currently broken for loading of old checkpoints, " \
                             "please revert to a previous checkpoint, e.g. 88095e7899497cf7a1da36fb6bbb6bc7b5370d53 " \
                             "until a fix is issued.")
        self.model.load_state_dict(state_dict)

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
        if self.cfg.mpc:
            return self.plan(obs, t0=t0, eval_mode=eval_mode, task=task).cpu()
        z = self.model.encode(obs, task)
        action, info = self.model.pi(z, task)
        if eval_mode:
            action = info["mean"]
        return action[0].cpu()
    
    @torch.no_grad()
    def act_traj(self, obs_traj, action_traj, t0=False, eval_mode=False, task=None):
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
        if task is not None:
            task = torch.tensor([task], device=self.device)
        if self.cfg.mpc:
            return self._plan_traj(obs_traj, action_traj, t0=t0, eval_mode=eval_mode, task=task).cpu()
        else:
            raise NotImplementedError('Not implemented for non-MPC mode')

    @torch.no_grad()
    def _estimate_value(self, obs, actions, task, sa_traj=False):
        """Estimate value of a trajectory starting at latent state z and executing given actions."""
        G, discount = 0, 1
        if sa_traj:
            observations, _ = self.model.next_traj(obs, actions, task, n_samples=self.cfg.num_samples, sa_traj=True)
        else:
            observations, _ = self.model.next_traj(obs, actions, task, n_samples=self.cfg.num_samples, atraj=True)
        observations = einops.rearrange(observations, 'batch horizon dim -> horizon batch dim')
        action_plan = actions[self.cfg.horizon:] if sa_traj else actions
        for t in range(self.cfg.horizon - 1):
            z = self.model.encode(observations[t], task)
            reward = math.two_hot_inv(self.model.reward(z, action_plan[t], task), self.cfg)
            G = G + discount * reward
            discount_update = self.discount[torch.tensor(task)] if self.cfg.multitask else self.discount
            discount = discount * discount_update
        # TODO: little problem, if i use traj here, it is not convinient to get next (s, a) for Q..
        # TODO: or tmp to use last (s, a) for Q, which sacrifices the last horizon & reward
        # TODO: currently choose the latter one
        z = self.model.encode(observations[-1], task)
        return G + discount * self.model.Q(z, action_plan[-1], task, return_type='avg')

    @torch.no_grad()
    def _plan(self, obs, t0=False, eval_mode=False, task=None):
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
        # whole_time = 0
        # Sample policy trajectories
        # t1 = time.time()
        if self.cfg.num_pi_trajs > 0:
            _obs = obs.repeat(self.cfg.num_pi_trajs, 1)
            _, pi_actions = self.model.next_traj(_obs, task, n_samples=self.cfg.num_pi_trajs)
            pi_actions = einops.rearrange(pi_actions, 'batch horizon dim -> horizon batch dim')
        # whole_time += time.time() - t1

        # Initialize state and parameters
        obs = obs.repeat(self.cfg.num_samples, 1)
        mean = torch.zeros(self.cfg.horizon, self.cfg.action_dim, device=self.device)
        std = torch.full((self.cfg.horizon, self.cfg.action_dim), self.cfg.max_std, dtype=torch.float, device=self.device)
        if not t0:
            mean[:-1] = self._prev_mean[1:]
        actions = torch.empty(self.cfg.horizon, self.cfg.num_samples, self.cfg.action_dim, device=self.device)
        if self.cfg.num_pi_trajs > 0:
            actions[:, :self.cfg.num_pi_trajs] = pi_actions

        # Iterate MPPI
        for _ in range(self.cfg.iterations):

            # Sample actions
            r = torch.randn(self.cfg.horizon, self.cfg.num_samples-self.cfg.num_pi_trajs, self.cfg.action_dim, device=std.device)
            actions_sample = mean.unsqueeze(1) + std.unsqueeze(1) * r
            actions_sample = actions_sample.clamp(-1, 1)
            actions[:, self.cfg.num_pi_trajs:] = actions_sample
            if self.cfg.multitask:
                actions = actions * self.model._action_masks[task]

            # Compute elite actions
            # t1 = time.time()
            value = self._estimate_value(obs, actions, task).nan_to_num(0)
            elite_idxs = torch.topk(value.squeeze(1), self.cfg.num_elites, dim=0).indices
            elite_value, elite_actions = value[elite_idxs], actions[:, elite_idxs]
            # whole_time += time.time() - t1

            # Update parameters
            max_value = elite_value.max(0).values
            score = torch.exp(self.cfg.temperature*(elite_value - max_value))
            score = score / score.sum(0)
            mean = (score.unsqueeze(0) * elite_actions).sum(dim=1) / (score.sum(0) + 1e-9)
            std = ((score.unsqueeze(0) * (elite_actions - mean.unsqueeze(1)) ** 2).sum(dim=1) / (score.sum(0) + 1e-9)).sqrt()
            std = std.clamp(self.cfg.min_std, self.cfg.max_std)
            if self.cfg.multitask:
                mean = mean * self.model._action_masks[task]
                std = std * self.model._action_masks[task]

        # Select action
        rand_idx = math.gumbel_softmax_sample(score.squeeze(1))  # gumbel_softmax_sample is compatible with cuda graphs
        actions = torch.index_select(elite_actions, 1, rand_idx).squeeze(1)
        a, std = actions[0], std[0]
        if not eval_mode:
            a = a + std * torch.randn(self.cfg.action_dim, device=std.device)
        self._prev_mean.copy_(mean)
        # print(f'>>>>> time for each diffusion:', whole_time / 7)
        # print('\n')
        return a.clamp(-1, 1)
    
    @torch.no_grad()
    def _plan_traj(self, obs_traj, action_traj, t0=False, eval_mode=False, task=None):
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
        # whole_time = 0
        # Sample policy trajectories
        # t1 = time.time()
        if self.cfg.num_pi_trajs > 0:
            obs_traj = torch.stack(obs_traj).to(self.device, non_blocking=True).unsqueeze(1)  # [horizon, 1, obs_dim]
            action_traj = torch.stack(action_traj).to(self.device, non_blocking=True).unsqueeze(1)  # [horizon, action_dim]
            _obs_traj = obs_traj.repeat(1, self.cfg.num_pi_trajs, 1)  # [horizon, num_pi_trajs, obs_dim]
            _action_traj = action_traj.repeat(1, self.cfg.num_pi_trajs, 1)
            _, pi_actions = self.model.next_traj(_obs_traj, _action_traj, task, n_samples=self.cfg.num_pi_trajs, sa_traj=True)
            pi_actions = einops.rearrange(pi_actions, 'batch horizon dim -> horizon batch dim')
        # whole_time += time.time() - t1

        # Initialize state and parameters
        obs_traj = obs_traj.repeat(1, self.cfg.num_samples, 1)  # [horizon, num_samples, obs_dim]
        action_traj = action_traj.repeat(1, self.cfg.num_samples, 1)
        mean = torch.zeros(self.cfg.horizon, self.cfg.action_dim, device=self.device)
        std = torch.full((self.cfg.horizon, self.cfg.action_dim), self.cfg.max_std, dtype=torch.float, device=self.device)
        if not t0:
            mean[:-1] = self._prev_mean[1:]
        actions = torch.empty(self.cfg.horizon, self.cfg.num_samples, self.cfg.action_dim, device=self.device)
        if self.cfg.num_pi_trajs > 0:
            actions[:, :self.cfg.num_pi_trajs] = pi_actions

        # Iterate MPPI
        for _ in range(self.cfg.iterations):

            # Sample actions
            r = torch.randn(self.cfg.horizon, self.cfg.num_samples-self.cfg.num_pi_trajs, self.cfg.action_dim, device=std.device)
            actions_sample = mean.unsqueeze(1) + std.unsqueeze(1) * r
            actions_sample = actions_sample.clamp(-1, 1)
            actions[:, self.cfg.num_pi_trajs:] = actions_sample
            if self.cfg.multitask:
                actions = actions * self.model._action_masks[task]

            # Compute elite actions
            # t1 = time.time()
            # TODO: sa_traj ver of _estimate_value, need obs_traj & action_traj & planned actions
            estimated_actions = torch.cat([action_traj, actions], dim=0)  # [horizon*2, num_samples, action_dim]
            value = self._estimate_value(obs_traj, estimated_actions, task, sa_traj=True).nan_to_num(0)
            elite_idxs = torch.topk(value.squeeze(1), self.cfg.num_elites, dim=0).indices
            elite_value, elite_actions = value[elite_idxs], actions[:, elite_idxs]
            # whole_time += time.time() - t1

            # Update parameters
            max_value = elite_value.max(0).values
            score = torch.exp(self.cfg.temperature*(elite_value - max_value))
            score = score / score.sum(0)
            mean = (score.unsqueeze(0) * elite_actions).sum(dim=1) / (score.sum(0) + 1e-9)
            std = ((score.unsqueeze(0) * (elite_actions - mean.unsqueeze(1)) ** 2).sum(dim=1) / (score.sum(0) + 1e-9)).sqrt()
            std = std.clamp(self.cfg.min_std, self.cfg.max_std)
            if self.cfg.multitask:
                mean = mean * self.model._action_masks[task]
                std = std * self.model._action_masks[task]

        # Select action
        rand_idx = math.gumbel_softmax_sample(score.squeeze(1))  # gumbel_softmax_sample is compatible with cuda graphs
        actions = torch.index_select(elite_actions, 1, rand_idx).squeeze(1)  # [horizon, action_dim]
        if not eval_mode:
            actions = actions + std * torch.randn(self.cfg.horizon, self.cfg.action_dim, device=std.device)
        self._prev_mean.copy_(mean)
        # print(f'>>>>> time for each diffusion:', whole_time / 7)
        # print('\n')
        return actions.clamp(-1, 1)


    def update_pi(self, zs, task):
        """
        Update policy using a sequence of latent states.

        Args:
            zs (torch.Tensor): Sequence of latent states.
            task (torch.Tensor): Task index (only used for multi-task experiments).

        Returns:
            float: Loss of the policy update.
        """
        action, info = self.model.pi(zs, task)
        qs = self.model.Q(zs, action, task, return_type='avg', detach=True)
        self.scale.update(qs[0])
        qs = self.scale(qs)

        # Loss is a weighted sum of Q-values
        rho = torch.pow(self.cfg.rho, torch.arange(len(qs), device=self.device))
        pi_loss = (-(self.cfg.entropy_coef * info["scaled_entropy"] + qs).mean(dim=(1,2)) * rho).mean()
        pi_loss.backward()
        pi_grad_norm = torch.nn.utils.clip_grad_norm_(self.model._pi.parameters(), self.cfg.grad_clip_norm)
        self.pi_optim.step()
        self.pi_optim.zero_grad(set_to_none=True)

        info = TensorDict({
            "pi_loss": pi_loss,
            "pi_grad_norm": pi_grad_norm,
            "pi_entropy": info["entropy"],
            "pi_scaled_entropy": info["scaled_entropy"],
            "pi_scale": self.scale.value,
        })
        return info

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
        action, _ = self.model.pi(next_z, task)
        discount = self.discount[task].unsqueeze(-1) if self.cfg.multitask else self.discount
        return reward + discount * self.model.Q(next_z, action, task, return_type='min', target=True)

    def _update(self, obs, action, reward, task=None):
        # Compute targets
        with torch.no_grad():
            next_z = self.model.encode(obs[1:], task)
            td_targets = self._td_target(next_z, reward, task)

        # Prepare for update
        self.model.train()

        # Latent rollout
        # print('>>>>>  Latent rollout....')
        consistency_loss = 0
        # TODO: how to use diffuser in training
        # TODO: origin stats in tdmpc2's shape[0] = horizon+1, here horizon
        states, _ = self.model.next_traj(obs[0], action, task, n_samples=self.cfg.num_pi_trajs, atraj=True)
        states = einops.rearrange(states, 'batch horizon dim -> horizon batch dim')
        for t, (_state, _next_obs) in enumerate(zip(states[1:].unbind(0), obs[1:].unbind(0))):
            consistency_loss = consistency_loss + F.mse_loss(_state, _next_obs) * self.cfg.rho**t

        # Predictions
        # print('>>>>>  Predictions....')
        zs = self.model.encode(states, task)
        _zs = zs[:-1]   # TODO: ?
        qs = self.model.Q(_zs, action, task, return_type='all')
        reward_preds = self.model.reward(_zs, action, task)

        # Compute losses
        reward_loss, value_loss = 0, 0
        for t, (rew_pred_unbind, rew_unbind, td_targets_unbind, qs_unbind) in enumerate(zip(reward_preds.unbind(0), reward.unbind(0), td_targets.unbind(0), qs.unbind(1))):
            reward_loss = reward_loss + math.soft_ce(rew_pred_unbind, rew_unbind, self.cfg).mean() * self.cfg.rho**t
            for _, qs_unbind_unbind in enumerate(qs_unbind.unbind(0)):
                value_loss = value_loss + math.soft_ce(qs_unbind_unbind, td_targets_unbind, self.cfg).mean() * self.cfg.rho**t

        consistency_loss = consistency_loss / self.cfg.horizon
        reward_loss = reward_loss / self.cfg.horizon
        value_loss = value_loss / (self.cfg.horizon * self.cfg.num_q)
        total_loss = (
            # self.cfg.consistency_coef * consistency_loss +
            self.cfg.reward_coef * reward_loss +
            self.cfg.value_coef * value_loss
        )

        # Update model
        # print('>>>>>  Updating model....')
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
        self.optim.step()
        self.optim.zero_grad(set_to_none=True)

        # Update policy
        pi_info = self.update_pi(zs.detach(), task)

        # Update target Q-functions
        self.model.soft_update_target_Q()

        # Return training statistics
        self.model.eval()
        info = TensorDict({
            "consistency_loss": consistency_loss,
            "reward_loss": reward_loss,
            "value_loss": value_loss,
            "total_loss": total_loss,
            "grad_norm": grad_norm,
        })
        print('=====  Finishing update: ')
        for k, v in info.items():
            print(f'=====       {k}: {v}')
        info.update(pi_info)
        return info.detach().mean()

    def update(self, buffer):
        """
        Main update function. Corresponds to one iteration of model learning.

        Args:
            buffer (common.buffer.Buffer): Replay buffer.

        Returns:
            dict: Dictionary of training statistics.
        """
        obs, action, reward, task = buffer.sample()
        kwargs = {}
        if task is not None:
            kwargs["task"] = task
        torch.compiler.cudagraph_mark_step_begin()
        return self._update(obs, action, reward, **kwargs)
