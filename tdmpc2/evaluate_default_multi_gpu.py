import xml.etree.ElementTree as ET
import os
import sys
os.environ['MUJOCO_GL'] = 'osmesa'
import warnings
warnings.filterwarnings('ignore')
os.environ['LAZY_LEGACY_OP'] = '0'

import re
import hydra
import imageio
import numpy as np
import torch
from termcolor import colored
from tqdm import tqdm

from common.parser import parse_cfg
from common.seed import set_seed
from envs import make_env
from tdmpc2 import TDMPC2, TDMPC2_MultiGPU
from accelerate import Accelerator
from common.logger import Logger


torch.backends.cudnn.benchmark = True

@hydra.main(config_name='config_default_eval_test', config_path='configs')
def evaluate_all_checkpoint(cfg: dict):
    def list_pt_files(directory):
        pt_files = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.pt'):
                    pt_files.append(os.path.join(root, file))
        return pt_files
    
    folder_path = '/root/tdmpc2/logs/mtgrab15/121/default-gpu-woeval/models'
    pt_files = list_pt_files(folder_path)
    assert torch.cuda.is_available()
    assert cfg.eval_episodes > 0, 'Must evaluate at least 1 episode.'
    cfg = parse_cfg(cfg)
    print(colored(f'Task: {cfg.task}', 'blue', attrs=['bold']))
    print(colored(f'Model size: {cfg.get("model_size", "default")}', 'blue', attrs=['bold']))
    print(colored(f'Checkpoint: {cfg.checkpoint}', 'blue', attrs=['bold']))
    
    accelerator = Accelerator()
    set_seed(cfg.seed + accelerator.process_index)
    # Make environment
    env = make_env(cfg)
    logger=Logger(cfg)
    for pt_file in pt_files:
        print(colored(f'Evaluating checkpoint: {pt_file}', 'blue', attrs=['bold']))
        evaluate(cfg, env, logger, accelerator, pt_file)
        break



def evaluate(cfg: dict, env, logger, accelerator, checkpoint_path=None):
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
    if checkpoint_path:
        cfg.checkpoint = checkpoint_path
    checkpoint_tag = re.findall(r'\d+', cfg.checkpoint)[-1]

    # Load agent
    agent = TDMPC2_MultiGPU(cfg, accelerator)
    assert os.path.exists(cfg.checkpoint), f'Checkpoint {cfg.checkpoint} not found! Must be a valid filepath.'
    agent.load(cfg.checkpoint)
    
    # Evaluate
    print(colored(f'Evaluating agent on {cfg.task}:', 'yellow', attrs=['bold']))
    if cfg.save_video:
        video_dir = os.path.join(cfg.work_dir, 'videos')
        os.makedirs(video_dir, exist_ok=True)
    results = dict()
    tasks = cfg.tasks if cfg.multitask else [cfg.task]
    for task_idx, task in enumerate(tasks):
        if cfg.task_name and (not task == cfg.task_name):
            continue  # single task
        # if not task_idx % 8 == accelerator.process_index:
        #     continue  # multi gpu, separate env
        
        overall_data = {}
        for key in ['reward', 'obs', 'task', 'action']:
            overall_data[key] = []
        
        ep_rewards, ep_successes = [], []
        for i in tqdm(range(cfg.eval_episodes), desc=f'gpu{accelerator.process_index}_{task}', unit="episode"):
            obs, done, ep_reward, t = env.reset(task_idx=task_idx), False, 0, 0
            
            if cfg.save_video:
                frames = [env.render()]
            while not done:
                action = agent.act(obs, t0=t==0, eval_mode=True, task=task_idx)
                obs, reward, done, info = env.step(action)
                ep_reward += reward
                t += 1
                if cfg.save_video:
                    frames.append(env.render())
            ep_rewards.append(ep_reward)
            ep_successes.append(info['success'])
            if cfg.save_video:
                if not os.path.exists(f'{video_dir}/{checkpoint_tag}'): os.mkdir(f'{video_dir}/{checkpoint_tag}')
                imageio.mimsave(
                    os.path.join(f'{video_dir}/{checkpoint_tag}', f'{task}-{i}.mp4'), frames, fps=15)
        print(f'gpu{accelerator.process_index}_{task}: {np.nanmean(ep_rewards):.2f}, {np.nanmean(ep_successes):.2f}, {t}')
        
        results.update({
            'iteration': int(checkpoint_tag),
            f'episode_reward+{cfg.tasks[task_idx]}': np.nanmean(ep_rewards),
            f'episode_success+{cfg.tasks[task_idx]}': np.nanmean(ep_successes),})

    logger.log(results, 'pretrain')






if __name__ == '__main__':
    evaluate_all_checkpoint()
