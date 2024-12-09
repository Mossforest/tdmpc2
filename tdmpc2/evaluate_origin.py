import os
os.environ['MUJOCO_GL'] = 'egl'
import warnings
warnings.filterwarnings('ignore')

import hydra
import imageio
import numpy as np
import torch
from termcolor import colored
import matplotlib.pyplot as plt

from common.parser import parse_cfg
from common.seed import set_seed
from envs import make_env
from tdmpc2 import TDMPC2

torch.backends.cudnn.benchmark = True

def make_dir(dir_path):
    """Create directory if it does not already exist."""
    try:
        os.makedirs(dir_path)
    except OSError:
        pass
    return dir_path

def plot_traj(data, name, task, path):
    plt.figure()
    plt.plot(data)

    # 添加标题和标签
    plt.title(f'task: {task}, {name}')
    plt.xlabel('timestep')
    plt.ylabel(name)

    # 显示图形
    plt.show()
    # 保存图形到文件
    path = make_dir(path / 'plot')
    plt.savefig(f'{path}/traj_eval_{name}.png')  # path: cfg.work_dir

@hydra.main(config_name='config_plan_origin', config_path='configs')
def evaluate(cfg: dict):
    """
    Script for evaluating a single-task / multi-task TD-MPC2 checkpoint.

    Most relevant args:
        `task`: task name (or mt30/mt80 for multi-task evaluation)
        `model_size`: model size, must be one of `[1, 5, 19, 48, 317]` (default: 5)
        `checkpoint`: path to model checkpoint to load
        `eval_episodes`: number of episodes to evaluate on per task (default: 10)
        `save_video`: whether to save a video of the evaluation (default: True)
        `seed`: random seed (default: 1)
    
    See config.yaml for a full list of args.

    Example usage:
    ````
        $ python evaluate.py task=mt80 model_size=48 checkpoint=/path/to/mt80-48M.pt
        $ python evaluate.py task=mt30 model_size=317 checkpoint=/path/to/mt30-317M.pt
        $ python evaluate.py task=dog-run checkpoint=/path/to/dog-1.pt save_video=true
    ```
    """
    assert torch.cuda.is_available()
    assert cfg.eval_episodes > 0, 'Must evaluate at least 1 episode.'
    cfg = parse_cfg(cfg)
    set_seed(cfg.seed)

    # Make environment
    env = make_env(cfg)

    # Load agent
    agent = TDMPC2(cfg)
    assert os.path.exists(cfg.checkpoint), f'Checkpoint {cfg.checkpoint} not found! Must be a valid filepath.'
    agent.load(cfg.checkpoint)
    print(colored(f'Checkpoint: {cfg.checkpoint}', 'blue', attrs=['bold']))
    
    # Evaluate
    print(colored(f'Evaluating agent on {cfg.task}, {cfg.sub_task}:', 'yellow', attrs=['bold']))
    scores = []
    ep_reward, ep_observation = [], []
    obs, done, reward, t = env.reset(), False, 0, 0
    ep_reward.append(reward)
    obs = obs / 100.0 * 2 - 1  # norm -> [-1, 1]
    obs[-1] = (obs[-1] + 1) / 7.0 - 1
    while not done:
        action = agent.act(obs, t0=t==0)
        obs, reward, done, info = env.step(action)
        obs = obs / 100.0 * 2 - 1  # norm -> [-1, 1]
        obs[-1] = (obs[-1] + 1) / 7.0 - 1
        ep_reward.append(reward)
        ep_observation.append(obs)
        t += 1
    ep_reward = np.array(ep_reward)
    ep_observation = torch.stack(ep_observation, dim=0).numpy().T
    mean_reward = np.mean(ep_reward)
    print(colored(f'  {cfg.task}, {cfg.sub_task}' \
        f'\tR: {mean_reward:.01f}  ', 'yellow'))
    # draw
    plot_traj(ep_reward, 'reward', f'{cfg.task}-{cfg.sub_task}', cfg.work_dir)
    plot_traj(ep_observation[0], 'obs_0', f'{cfg.task}-{cfg.sub_task}', cfg.work_dir)
    plot_traj(ep_observation[1], 'obs_1', f'{cfg.task}-{cfg.sub_task}', cfg.work_dir)
    plot_traj(ep_observation[2], 'obs_2', f'{cfg.task}-{cfg.sub_task}', cfg.work_dir)
    plot_traj(ep_observation[3], 'obs_3', f'{cfg.task}-{cfg.sub_task}', cfg.work_dir)
    plot_traj(ep_observation[4], 'obs_4', f'{cfg.task}-{cfg.sub_task}', cfg.work_dir)
    plot_traj(ep_observation[5], 'obs_5', f'{cfg.task}-{cfg.sub_task}', cfg.work_dir)



if __name__ == '__main__':
    evaluate()
