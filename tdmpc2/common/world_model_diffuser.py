from copy import deepcopy
import einops

import numpy as np
import torch
import torch.nn as nn

from common import layers, math, init
from tensordict import TensorDict
from tensordict.nn import TensorDictParams
from diffuser.utils import load_diffusion
from common.init import to_torch, to_np

class WorldModelDiffuser(nn.Module):
    """
    TD-MPC2 implicit world model architecture.
    Can be used for both single-task and multi-task experiments.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self._encoder = layers.enc(cfg)
        # self._dynamics = layers.mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim, 2*[cfg.mlp_dim], cfg.latent_dim, act=layers.SimNorm(cfg))
        self._reward = layers.mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim, 2*[cfg.mlp_dim], max(cfg.num_bins, 1))
        # self._pi = layers.mlp(cfg.latent_dim + cfg.task_dim, 2*[cfg.mlp_dim], 2*cfg.action_dim)
        self._Qs = layers.Ensemble([layers.mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim, 2*[cfg.mlp_dim], max(cfg.num_bins, 1), dropout=cfg.dropout) for _ in range(cfg.num_q)])
        self.apply(init.weight_init)
        init.zero_([self._reward[-1].weight, self._Qs.params["2", "weight"]])
        self._dynamics = self.load_diffuser(cfg.diffuser_pretrained_path)

        self.register_buffer("log_std_min", torch.tensor(cfg.log_std_min))
        self.register_buffer("log_std_dif", torch.tensor(cfg.log_std_max) - self.log_std_min)
        self.init()

    def init(self):
        # Create params
        self._detach_Qs_params = TensorDictParams(self._Qs.params.data, no_convert=True)
        self._target_Qs_params = TensorDictParams(self._Qs.params.data.clone(), no_convert=True)

        # Create modules
        with self._detach_Qs_params.data.to("meta").to_module(self._Qs.module):
            self._detach_Qs = deepcopy(self._Qs)
            self._target_Qs = deepcopy(self._Qs)

        # Assign params to modules
        self._detach_Qs.params = self._detach_Qs_params
        self._target_Qs.params = self._target_Qs_params

    def __repr__(self):
        repr = 'TD-MPC2 World Model\n'
        modules = ['Encoder', 'Dynamics', 'Reward', 'Q-functions']
        for i, m in enumerate([self._encoder, self._dynamics, self._reward, self._Qs]):
            repr += f"{modules[i]}: {m}\n"
        repr += "Learnable parameters: {:,}".format(self.total_params)
        return repr

    def load_diffuser(self, path):
        diffusion_experiment = load_diffusion(path)
        dataset = diffusion_experiment.dataset
        # renderer = diffusion_experiment.renderer
        model = diffusion_experiment.trainer.ema_model
        self._diffuser_dataset = dataset
        return model

    @property
    def total_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.init()
        return self

    def train(self, mode=True):
        """
        Overriding `train` method to keep target Q-networks in eval mode.
        """
        super().train(mode)
        self._target_Qs.train(False)
        return self

    def soft_update_target_Q(self):
        """
        Soft-update target Q-networks using Polyak averaging.
        """
        self._target_Qs_params.lerp_(self._detach_Qs_params, self.cfg.tau)

    def task_emb(self, x, task):
        """
        Continuous task embedding for multi-task experiments.
        Retrieves the task embedding for a given task ID `task`
        and concatenates it to the input `x`.
        """
        if isinstance(task, int):
            task = torch.tensor([task], device=x.device)
        emb = self._task_emb(task.long())
        if x.ndim == 3:
            emb = emb.unsqueeze(0).repeat(x.shape[0], 1, 1)
        elif emb.shape[0] == 1:
            emb = emb.repeat(x.shape[0], 1)
        return torch.cat([x, emb], dim=-1)

    def encode(self, obs, task):
        """
        Encodes an observation into its latent representation.
        This implementation assumes a single state-based observation.
        """
        if self.cfg.multitask:
            obs = self.task_emb(obs, task)
        if self.cfg.obs == 'rgb' and obs.ndim == 5:
            return torch.stack([self._encoder[self.cfg.obs](o) for o in obs])
        return self._encoder[self.cfg.obs](obs)
    
    def run_diffusion(self, obs, action, n_samples=1, device='cuda:0', need_action=True, **diffusion_kwargs):
        ## normalize observation for model
        obs_np = to_np(obs)
        action_np = to_np(action)
        obs_np = self._diffuser_dataset.normalizer.normalize(obs_np, 'observations')
        action_np = self._diffuser_dataset.normalizer.normalize(action_np, 'actions')

        # ## add a batch dimension and repeat for multiple samples
        # ## [ observation_dim ] --> [ n_samples x observation_dim ]
        # obs = obs[None].repeat(n_samples, axis=0)
        # action = action[None].repeat(n_samples, axis=0)

        ## format `conditions` input for model
        conditions = {
            0: tuple([to_torch(obs_np, device=device), to_torch(action_np, device=device)])
        }

        samples = self._dynamics.conditional_sample_sa(conditions, n_samples=n_samples,
                horizon=self.cfg.horizon, return_chain=True, verbose=False, **diffusion_kwargs)
        diffusion = samples.chains

        ## [ n_samples x (n_diffusion_steps + 1) x horizon x (action_dim + observation_dim)]
        diffusion = to_np(diffusion)

        ## extract observations
        ## [ n_samples x (n_diffusion_steps + 1) x horizon x observation_dim ]
        normed_observations = diffusion[:, :, :, self._diffuser_dataset.action_dim:]
        observations = self._diffuser_dataset.normalizer.unnormalize(normed_observations, 'observations')
        ## [ (n_diffusion_steps + 1) x n_samples x horizon x observation_dim ]
        observations = einops.rearrange(observations,
                                        'batch steps horizon dim -> steps batch horizon dim')
        
        if need_action:
            normed_actions = diffusion[:, :, :, :self._diffuser_dataset.action_dim]
            actions = self._diffuser_dataset.normalizer.unnormalize(normed_actions, 'actions')
            actions = einops.rearrange(actions,
                                    'batch steps horizon dim -> steps batch horizon dim')
            
            return observations, actions

        return observations
    
    
    def run_diffusion_wo_action(self, obs, n_samples=1, device='cuda:0', **diffusion_kwargs):
        ## normalize observation for model
        obs_np = to_np(obs)
        obs_np = self._diffuser_dataset.normalizer.normalize(obs_np, 'observations')

        ## format `conditions` input for model
        conditions = {
            0: to_torch(obs_np, device=device)
        }

        samples = self._dynamics.conditional_sample_wo_action(conditions, n_samples=n_samples,
                horizon=self.cfg.horizon, return_chain=True, verbose=False, **diffusion_kwargs)
        diffusion = samples.chains

        ## [ n_samples x (n_diffusion_steps + 1) x horizon x (action_dim + observation_dim)]
        diffusion = to_np(diffusion)

        ## extract observations
        ## [ n_samples x (n_diffusion_steps + 1) x horizon x observation_dim ]
        normed_observations = diffusion[:, :, :, self._diffuser_dataset.action_dim:]
        observations = self._diffuser_dataset.normalizer.unnormalize(normed_observations, 'observations')
        ## [ (n_diffusion_steps + 1) x n_samples x horizon x observation_dim ]
        observations = einops.rearrange(observations,
                                        'batch steps horizon dim -> steps batch horizon dim')
        
        normed_actions = diffusion[:, :, :, :self._diffuser_dataset.action_dim]
        actions = self._diffuser_dataset.normalizer.unnormalize(normed_actions, 'actions')
        actions = einops.rearrange(actions,
                                  'batch steps horizon dim -> steps batch horizon dim')

        return observations, actions

    def run_diffusion_action_traj(self, obs, action_traj, n_samples=1, device='cuda:0', need_action=True, **diffusion_kwargs):
        ## normalize observation for model
        obs_np = to_np(obs)
        action_np = to_np(action_traj)
        obs_np = self._diffuser_dataset.normalizer.normalize(obs_np, 'observations')
        action_np = self._diffuser_dataset.normalizer.normalize(action_np, 'actions')

        ## format `conditions` input for model
        conditions = {
            0: tuple([to_torch(obs_np, device=device), to_torch(action_np[0], device=device)])
        }
        for i in range(1, action_traj.shape[0]):
            conditions[i] = tuple([None, to_torch(action_np[i], device=device)])

        samples = self._dynamics.conditional_sample_atraj(conditions, n_samples=n_samples,
                horizon=self.cfg.horizon, return_chain=True, verbose=False, **diffusion_kwargs)
        diffusion = samples.chains

        ## [ n_samples x (n_diffusion_steps + 1) x horizon x (action_dim + observation_dim)]
        diffusion = to_np(diffusion)

        ## extract observations
        ## [ n_samples x (n_diffusion_steps + 1) x horizon x observation_dim ]
        normed_observations = diffusion[:, :, :, self._diffuser_dataset.action_dim:]
        observations = self._diffuser_dataset.normalizer.unnormalize(normed_observations, 'observations')
        ## [ (n_diffusion_steps + 1) x n_samples x horizon x observation_dim ]
        observations = einops.rearrange(observations,
                                        'batch steps horizon dim -> steps batch horizon dim')
        
        if need_action:
            normed_actions = diffusion[:, :, :, :self._diffuser_dataset.action_dim]
            actions = self._diffuser_dataset.normalizer.unnormalize(normed_actions, 'actions')
            actions = einops.rearrange(actions,
                                    'batch steps horizon dim -> steps batch horizon dim')
            
            return observations, actions

        return observations
    
    def run_diffusion_sa_traj(self, obs_traj, action_traj, n_samples=1, device='cuda:0', need_action=True, **diffusion_kwargs):
        ## normalize observation for model
        obs_np = to_np(obs_traj)  # [horizon+1, n_samples, obs_dim]
        action_np = to_np(action_traj)
        obs_np = self._diffuser_dataset.normalizer.normalize(obs_np, 'observations')
        action_np = self._diffuser_dataset.normalizer.normalize(action_np, 'actions')

        ## format `conditions` input for model
        conditions = {}
        for i in range(max(len(obs_np), len(action_np))):
            if i >= len(obs_np):
                s = None
            else:
                s = to_torch(obs_np[i], device=device) if not np.isnan(obs_np[i]).any() else None
            if i >= len(action_np):
                a = None
            else:
                a = to_torch(action_np[i], device=device) if not np.isnan(action_np[i]).any() else None
            conditions[i] = tuple([s, a])

        samples = self._dynamics.conditional_sample_sa_traj(conditions, n_samples=n_samples,
                horizon=self.cfg.horizon*2, return_chain=True, verbose=False, **diffusion_kwargs)
        diffusion = samples.chains

        ## [ n_samples x (n_diffusion_steps + 1) x horizon x (action_dim + observation_dim)]
        diffusion = to_np(diffusion)[:, :, self.cfg.horizon:]  # from horizon*2 to horizon

        ## extract observations
        ## [ n_samples x (n_diffusion_steps + 1) x horizon x observation_dim ]
        normed_observations = diffusion[:, :, :, self._diffuser_dataset.action_dim:]
        observations = self._diffuser_dataset.normalizer.unnormalize(normed_observations, 'observations')
        ## [ (n_diffusion_steps + 1) x n_samples x horizon x observation_dim ]
        observations = einops.rearrange(observations,
                                        'batch steps horizon dim -> steps batch horizon dim')
        
        if need_action:
            normed_actions = diffusion[:, :, :, :self._diffuser_dataset.action_dim]
            actions = self._diffuser_dataset.normalizer.unnormalize(normed_actions, 'actions')
            actions = einops.rearrange(actions,
                                    'batch steps horizon dim -> steps batch horizon dim')
            
            return observations, actions

        return observations

    def run_diffusion_pi(self, obs, n_samples=1, device='cuda:0', **diffusion_kwargs):
        # this function only takes one obs, horizon=1, diffusion on [a, s] to get a = pi(s)
        # obs: [n_samples, horizon=1, obs_dim]
        ## normalize observation for model
        obs_np = to_np(obs)
        obs_np = self._diffuser_dataset.normalizer.normalize(obs_np, 'observations')

        ## format `conditions` input for model
        conditions = {
            0: to_torch(obs_np, device=device)
        }

        # default conditional_sample function
        samples = self._dynamics.conditional_sample_wo_action(conditions, n_samples=n_samples,
                horizon=1, return_chain=True, verbose=False, **diffusion_kwargs)
        diffusion = samples.chains

        ## [ n_samples x (n_diffusion_steps + 1) x horizon=1 x (action_dim + observation_dim)]
        diffusion = to_np(diffusion)

        ## extract observations
        ## [ n_samples x (n_diffusion_steps + 1) x horizon=1 x observation_dim ]
        normed_observations = diffusion[:, :, :, self._diffuser_dataset.action_dim:]
        observations = self._diffuser_dataset.normalizer.unnormalize(normed_observations, 'observations')
        ## [ (n_diffusion_steps + 1) x n_samples x horizon=1 x observation_dim ]
        observations = einops.rearrange(observations,
                                        'batch steps horizon dim -> steps batch horizon dim')
        
        normed_actions = diffusion[:, :, :, :self._diffuser_dataset.action_dim]
        actions = self._diffuser_dataset.normalizer.unnormalize(normed_actions, 'actions')
        actions = einops.rearrange(actions,
                                  'batch steps horizon dim -> steps batch horizon dim')

        return actions

    def next_traj(self, obs, task=None, n_samples=1):
        # obs shape: [num_samples, obs_dim]
        observations, actions = self.run_diffusion_wo_action(obs, n_samples)
        observations = to_torch(observations[-1], device=obs.device)   # [n_samples, horizon, 11]
        actions = to_torch(actions[-1], device=obs.device)   # [n_samples, horizon, 11]
        return observations, actions

    def next(self, obs, a, task=None, n_samples=1):
        """
        Predicts the next latent state given the current latent state and action.
        """
        # [n_samples, 11], tmp to discard all traj horizon after next_obs
        observations = self.run_diffusion(obs, a, n_samples, need_action=False)   # [21, n_samples, horizon, 11]
        observations = observations[-1]   # [n_samples, horizon, 11]
        observations = observations[:, 0, :]   # [n_samples, 11]
        return to_torch(observations, device=obs.device)
    
    def diffusion_loss(self, obs, action, task=None):
        # obs shape: torch.tensor, [horizon, num_samples, obs_dim]
        # action the same
        obs_ = torch.nan_to_num(obs, nan=0.0)
        action_ = torch.nan_to_num(action, nan=0.0)
        device = obs.device
        obs_np = to_np(obs_)
        action_np = to_np(action_)
        obs_np = self._diffuser_dataset.normalizer.normalize(obs_np, 'observations')
        action_np = self._diffuser_dataset.normalizer.normalize(action_np, 'actions')
        traj_np = np.concatenate([action_np, obs_np], axis=-1)
        trajectories = to_torch(traj_np, device=device)
        trajectories = trajectories.permute(1, 0, 2)   # [num_samples, horizon, action_dim+obs_dim]

        conditions = {
            0: to_torch(obs_np[0], device=device)
        }
        
        loss = self._dynamics.loss(trajectories, conditions)
        return loss
    
    def pi_next(self, obs, task=None, n_samples=1):
        """
        Predicts the next latent state given the current latent state and action.
        """
        # obs: [n_samples,  obs_dim]
        actions = self.run_diffusion_pi(obs, n_samples)   # [21, n_samples, horizon=1, a_dim]
        actions = actions[-1].squeeze()   # [n_samples, 11]
        return to_torch(actions, device=obs.device)

    def reward(self, z, a, task=None):
        """
        Predicts instantaneous (single-step) reward.
        """
        z = torch.cat([z, a], dim=-1)
        return self._reward(z)

    def Q(self, z, a, task, return_type='min', target=False, detach=False):
        """
        Predict state-action value.
        `return_type` can be one of [`min`, `avg`, `all`]:
            - `min`: return the minimum of two randomly subsampled Q-values.
            - `avg`: return the average of two randomly subsampled Q-values.
            - `all`: return all Q-values.
        `target` specifies whether to use the target Q-networks or not.
        """
        assert return_type in {'min', 'avg', 'all'}

        if self.cfg.multitask:
            z = self.task_emb(z, task)

        z = torch.cat([z, a], dim=-1)
        if target:
            qnet = self._target_Qs
        elif detach:
            qnet = self._detach_Qs
        else:
            qnet = self._Qs
        out = qnet(z)

        if return_type == 'all':
            return out

        qidx = torch.randperm(self.cfg.num_q, device=out.device)[:2]
        Q = math.two_hot_inv(out[qidx], self.cfg)
        if return_type == "min":
            return Q.min(0).values
        return Q.sum(0) / 2