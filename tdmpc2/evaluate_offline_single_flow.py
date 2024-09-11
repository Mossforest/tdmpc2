import xml.etree.ElementTree as ET
import os
os.environ['MUJOCO_GL'] = 'egl'
import warnings
warnings.filterwarnings('ignore')

import hydra
import time
import imageio
import numpy as np
import torch
import torch.nn.functional as F
from termcolor import colored
from tensordict import TensorDict
from tqdm import tqdm

from common.parser import parse_cfg
from common.seed import set_seed
from envs import make_env
from tdmpc2 import TDMPC2_Flow
from common import TASK_SET

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

torch.backends.cudnn.benchmark = True

def convert_task_idx(task_name, dest_task_list):
    return dest_task_list.index(task_name)


def evaluate(cfg: dict, friction):
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
    print(colored(f'Task: {cfg.task}', 'blue', attrs=['bold']))
    print(colored(f'Model size: {cfg.get("model_size", "default")}', 'blue', attrs=['bold']))
    print(colored(f'Checkpoint: {cfg.checkpoint}', 'blue', attrs=['bold']))

    # Make environment
    env = make_env(cfg)

    # Load agent
    agent = TDMPC2_Flow(cfg)
    assert os.path.exists(cfg.checkpoint), f'Checkpoint {cfg.checkpoint} not found! Must be a valid filepath.'
    agent.load(cfg.checkpoint)
    
    # Evaluate
    print(colored(f'Evaluating agent on {cfg.task}:', 'yellow', attrs=['bold']))
    if cfg.save_video:
        video_dir = os.path.join(cfg.work_dir, 'videos')
        os.makedirs(video_dir, exist_ok=True)
    scores = []
    tasks = cfg.tasks if cfg.multitask else [cfg.task]
    eval_tasks = TASK_SET.get(cfg.eval_task, [cfg.eval_task])
    for task_idx, task in enumerate(tasks):
        if cfg.task_name and (not task == cfg.task_name):
            continue  # single task
        if not task in eval_tasks:
            continue  # skip the multi-task not want to collect
        
        overall_data = {}
        for key in ['reward', 'obs', 'task', 'action']:
            overall_data[key] = []
        
        eval_task_idx = convert_task_idx(task, eval_tasks)
        ep_rewards, ep_successes = [], []
        for i in tqdm(range(cfg.eval_episodes), desc=task, unit="episode"):
            obs, done, ep_reward, t = env.reset(task_idx=task_idx), False, 0, 0
            
            if cfg.save_dynamics:
                collected_data = {}
                for key in ['reward', 'obs', 'task', 'action']:
                    collected_data[key] = []
                collected_data['reward'].append(ep_reward)
                collected_data['obs'].append(obs)
                collected_data['task'].append(eval_task_idx)
            
            if cfg.save_video:
                frames = [env.render()]
            while not done:
                t1 = time.time()
                action = agent.act(obs, t0=t==0, task=task_idx)
                if cfg.save_dynamics:
                    collected_data['action'].append(action[:4])   # ！: last tuple
                
                obs, reward, done, info = env.step(action)
                if cfg.save_dynamics:
                    collected_data['reward'].append(ep_reward)
                    collected_data['obs'].append(obs)
                    collected_data['task'].append(eval_task_idx)
                ep_reward += reward
                t += 1
                if cfg.save_video:
                    frames.append(env.render())
                print(f'step {t}: time comsumed {time.time() - t1}')
            # print('='*20, f'episoode{i} finished, time comsuming {(time.time() - t1):.2f}\n\n')    268.87s
            if cfg.save_dynamics:
                collected_data['action'].append(torch.zeros_like(action[:4]))   # ！: last tuple
            ep_rewards.append(ep_reward)
            ep_successes.append(info['success'])
            if cfg.save_video:
                if not os.path.exists(f'{video_dir}/{friction}'): os.mkdir(f'{video_dir}/{friction}')
                imageio.mimsave(
                    os.path.join(f'{video_dir}/{friction}', f'{task}-{i}.mp4'), frames, fps=15)
            
            if cfg.save_dynamics:
                for key in ['reward', 'obs', 'task', 'action']:
                    if isinstance(collected_data[key][0], torch.Tensor):
                        collected_data[key] = torch.stack(collected_data[key])
                    else:
                        collected_data[key] = torch.tensor(collected_data[key])
                
                
                pad_length = 101 - collected_data['task'].shape[0]  # mostly 101
                if pad_length > 0:
                    # 使用pad函数在向量的末尾填充零
                    for key in ['reward', 'obs', 'task', 'action']:
                        collected_data[key] = F.pad(collected_data[key], (0, pad_length), "constant", 0)
                elif pad_length < 0:
                    for key in ['reward', 'obs', 'task', 'action']:
                        collected_data[key] = collected_data[key][:101]
                
                for key in ['reward', 'obs', 'task', 'action']:
                    overall_data[key].append(collected_data[key])
        
        if cfg.save_dynamics:
            overall_tensordict = TensorDict()
            for key in ['reward', 'obs', 'task', 'action']:
                overall_tensordict[key] = torch.stack(overall_data[key])
            overall_tensordict.shape = overall_tensordict['task'].shape
            if not os.path.exists(cfg.dynamic_save_path): os.mkdir(cfg.dynamic_save_path)
            if not os.path.exists(f'{cfg.dynamic_save_path}/{friction}'): os.mkdir(f'{cfg.dynamic_save_path}/{friction}')
            torch.save(overall_tensordict, f'{cfg.dynamic_save_path}/{friction}/{task}.pt')
        
        ep_rewards = np.mean(ep_rewards)
        ep_successes = np.mean(ep_successes)
        if cfg.multitask:
            scores.append(ep_successes*100 if task.startswith('mw-') else ep_rewards/10)
        print(colored(f'  {task:<22}' \
            f'\tR: {ep_rewards:.01f}  ' \
            f'\tS: {ep_successes:.02f}', 'yellow'))
    if cfg.multitask:
        print(colored(f'Normalized score: {np.mean(scores):.02f}', 'yellow', attrs=['bold']))





@hydra.main(config_name='config_evaluate_flow_gnn', config_path='configs')
def edit_and_evaluate(cfg: dict):

    # XML文件路径
    xml_file_path = cfg.metaworld_path + '/metaworld/envs/assets_v2/objects/assets/xyz_base.xml'
    frictions = [str(fric)+" 0.1 0.002" for fric in cfg.friction]
    for idx, friction in enumerate(frictions):
        # 解析XML文件
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        for geom in root.findall('.//geom'):
            # 检查geom元素是否有name属性且值为'leftpad_geom'
            if 'name' in geom.attrib and geom.get('name') in ['leftpad_geom', 'rightpad_geom']:
                # 如果friction属性存在，则更改它
                if 'friction' in geom.attrib:
                    geom.set('friction', friction)
        # 保存修改后的XML文件
        tree.write(xml_file_path)
        
        evaluate(cfg, cfg.friction[idx])
        print(f'\n\n ========== done: friction {cfg.friction[idx]} ==========')




if __name__ == '__main__':
    edit_and_evaluate()
