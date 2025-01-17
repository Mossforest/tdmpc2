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

    def common_metrics(self):
        """Return a dictionary of current metrics."""
        return dict(
            step=self._step,
            episode=self._ep_idx,
            total_time=time() - self._start_time,
        )

    def eval_traj(self):
        """Evaluate a TD-MPC2 agent."""
        ep_rewards, ep_successes = [], []
        for i in range(self.cfg.eval_episodes):
            obs, done, ep_reward, t = self.env.reset(), False, 0, 0
            action_plan = [self.env.rand_act()] * self.cfg.horizon
            observation_traj = [obs] * self.cfg.horizon  # TODO: start, use the only one observation
            action_ptr = self.cfg.horizon
            if self.cfg.save_video:
                self.logger.video.init(self.env, enabled=(i==0))
            t1 = time()
            while not done:
                torch.compiler.cudagraph_mark_step_begin()
                # t2 = time()
                if action_ptr >= self.cfg.horizon:
                    assert len(observation_traj) == self.cfg.horizon  # history length
                    action_plan = self.agent.act_traj(observation_traj, action_plan, t0=t==0, eval_mode=True)
                    action_plan = [a for a in action_plan]
                    assert len(action_plan) == self.cfg.horizon  # planning length
                    observation_traj.clear()
                    action_ptr = 0
                # print(f'          >>>>> Time taken for one step:', time() - t2)
                obs, reward, done, info = self.env.step(action_plan[action_ptr])
                observation_traj.append(obs)
                ep_reward += reward
                t += 1
                action_ptr += 1
                if self.cfg.save_video:
                    self.logger.video.record(self.env)
                if t % 10 == 0:
                    print('>>>>> Current step:', t)
            ep_rewards.append(ep_reward)
            ep_successes.append(info['success'])
            if self.cfg.save_video:
                self.logger.video.save(self._step)
            print('>>>>> Time taken for one episode in {t} step:', time() - t1)
            print('>>>>> Episode', i, 'reward:', ep_reward)
            exit()
        return dict(
            episode_reward=np.nanmean(ep_rewards),
            episode_success=np.nanmean(ep_successes),
        )


    def eval(self):
        """Evaluate a TD-MPC2 agent."""
        ep_rewards, ep_successes = [], []
        for i in range(self.cfg.eval_episodes):
            obs, done, ep_reward, t = self.env.reset(), False, 0, 0
            if self.cfg.save_video:
                self.logger.video.init(self.env, enabled=(i==0))
            t1 = time()
            while not done:  # todo: and t < self.cfg.episode_length:
                torch.compiler.cudagraph_mark_step_begin()
                # t2 = time()
                action = self.agent.act(obs, t0=t==0, eval_mode=True)
                # print(f'          >>>>> Time taken for one step:', time() - t2)
                obs, reward, done, info = self.env.step(action)
                ep_reward += reward
                t += 1
                if self.cfg.save_video:
                    self.logger.video.record(self.env)
                if t % 10 == 0:
                    print('>>>>> Current step:', t)
            ep_rewards.append(ep_reward)
            ep_successes.append(info['success'])
            if self.cfg.save_video:
                self.logger.video.save(self._step)
            print('>>>>> Time taken for one episode in {t} step:', time() - t1)
            print('>>>>> Episode', i, 'reward:', ep_reward)
            exit()
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
        td = TensorDict(
            obs=obs,
            action=action.unsqueeze(0),
            reward=reward.unsqueeze(0),
        batch_size=(1,))
        return td

    def train(self):
        """Train a TD-MPC2 agent."""
        train_metrics, done, eval_next = {}, True, False
        while self._step <= self.cfg.steps:
            # Evaluate agent periodically
            if self._step % self.cfg.eval_freq == 0:
                eval_next = True

            # Reset environment
            if done:
                if eval_next:
                    # print('>>>>>  Evaluating....')
                    eval_metrics = self.eval_traj()  # self.eval()
                    eval_metrics.update(self.common_metrics())
                    self.logger.log(eval_metrics, 'eval')
                    eval_next = False

                if self._step > 0:
                    train_metrics.update(
                        episode_reward=torch.tensor([td['reward'] for td in self._tds[1:]]).sum(),
                        episode_success=info['success'],
                    )
                    train_metrics.update(self.common_metrics())
                    self.logger.log(train_metrics, 'train')
                    self._ep_idx = self.buffer.add(torch.cat(self._tds))

                obs = self.env.reset()
                self._tds = [self.to_td(obs)]

            # Collect experience
            # print('>>>>>  Collecting data....')
            if self._step > self.cfg.seed_steps:
                action = self.agent.act(obs, t0=len(self._tds)==1)
            else:
                action = self.env.rand_act()
            obs, reward, done, info = self.env.step(action)
            self._tds.append(self.to_td(obs, action, reward))

            # Update agent
            # print('>>>>>  Updating agent....')
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
