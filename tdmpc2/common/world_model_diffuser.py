from copy import deepcopy
import einops

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
        self._dynamics = self.load_diffuser(cfg.diffuser_pretrained_path)
        self._reward = layers.mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim, 2*[cfg.mlp_dim], max(cfg.num_bins, 1))
        self._pi = layers.mlp(cfg.latent_dim + cfg.task_dim, 2*[cfg.mlp_dim], 2*cfg.action_dim)
        self._Qs = layers.Ensemble([layers.mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim, 2*[cfg.mlp_dim], max(cfg.num_bins, 1), dropout=cfg.dropout) for _ in range(cfg.num_q)])
        self.apply(init.weight_init)
        init.zero_([self._reward[-1].weight, self._Qs.params["2", "weight"]])

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
        modules = ['Encoder', 'Dynamics', 'Reward', 'Policy prior', 'Q-functions']
        for i, m in enumerate([self._encoder, self._dynamics, self._reward, self._pi, self._Qs]):
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
    
    def run_diffusion(self, obs, action, n_samples=1, device='cuda:0', **diffusion_kwargs):
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

        ## unnormalize observation samples from model
        observations = self._diffuser_dataset.normalizer.unnormalize(normed_observations, 'observations')

        ## [ (n_diffusion_steps + 1) x n_samples x horizon x observation_dim ]
        observations = einops.rearrange(observations,
                                        'batch steps horizon dim -> steps batch horizon dim')

        return observations
    
    def diffusion_next(self, obs, action, n_samples=1, device='cuda:0', **diffusion_kwargs):
        diffusion_chain = self.run_diffusion(obs, action, n_samples, device, **diffusion_kwargs)   # [21, n_samples, 32, 11]
        observations = diffusion_chain[-1]   # [n_samples, 32, 11]
        next_obs = observations[:, 0, :]   # todo: [n_samples, 11], tmp to discard all traj timestep after next_obs
        return to_torch(next_obs, device=device)

    def next(self, obs, a, task=None, n_samples=1):
        """
        Predicts the next latent state given the current latent state and action.
        """
        return self.diffusion_next(obs, a, n_samples)

    def reward(self, z, a, task=None):
        """
        Predicts instantaneous (single-step) reward.
        """
        z = torch.cat([z, a], dim=-1)
        return self._reward(z)

    def pi(self, z, task):
        """
        Samples an action from the policy prior.
        The policy prior is a Gaussian distribution with
        mean and (log) std predicted by a neural network.
        """
        if self.cfg.multitask:
            z = self.task_emb(z, task)

        # Gaussian policy prior
        mean, log_std = self._pi(z).chunk(2, dim=-1)
        log_std = math.log_std(log_std, self.log_std_min, self.log_std_dif)
        eps = torch.randn_like(mean)

        if self.cfg.multitask: # Mask out unused action dimensions
            mean = mean * self._action_masks[task]
            log_std = log_std * self._action_masks[task]
            eps = eps * self._action_masks[task]
            action_dims = self._action_masks.sum(-1)[task].unsqueeze(-1)
        else: # No masking
            action_dims = None

        log_prob = math.gaussian_logprob(eps, log_std)

        # Scale log probability by action dimensions
        size = eps.shape[-1] if action_dims is None else action_dims
        scaled_log_prob = log_prob * size

        # Reparameterization trick
        action = mean + eps * log_std.exp()
        mean, action, log_prob = math.squash(mean, action, log_prob)

        entropy_scale = scaled_log_prob / (log_prob + 1e-8)
        info = TensorDict({
            "mean": mean,
            "log_std": log_std,
            "action_prob": 1.,
            "entropy": -log_prob,
            "scaled_entropy": -log_prob * entropy_scale,
        })
        return action, info

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