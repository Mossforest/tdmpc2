import os
os.environ['MUJOCO_GL'] = 'egl'
import warnings
warnings.filterwarnings('ignore')

import hydra
import pickle
import imageio
import numpy as np
import torch
from termcolor import colored
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from common.parser import parse_cfg
from common.seed import set_seed
from envs import make_env
from tdmpc2 import TDMPC2_Flow

torch.backends.cudnn.benchmark = True

def make_dir(dir_path):
    """Create directory if it does not already exist."""
    try:
        os.makedirs(dir_path)
    except OSError:
        pass
    return dir_path

def apply_pca_parameters(data_2, pca_parameters):
    """
    使用保存的PCA参数对新的n维数据进行降维。
    
    参数:
    data_2 : numpy.ndarray
        新的待降维的数据，形状为 (n_samples, n_features)。
    pca_parameters : dict
        之前保存的PCA模型参数，包括mean、components_和explained_variance_。
    
    返回:
    reduced_data_2 : numpy.ndarray
        使用保存的PCA参数降维后的数据。
    """
    # 提取PCA参数
    mean = pca_parameters['mean']
    components = pca_parameters['components_']
    explained_variance = pca_parameters['explained_variance_']

    # 标准化新的数据
    scaler = StandardScaler()
    scaler.mean_ = mean  # 使用保存的均值
    scaler.scale_ = [np.ones(len(mean)) * np.std(data_2, axis=0)]  # 假设标准差不变
    data_2_normalized = scaler.transform(data_2)
    
    # 创建一个新的PCA对象，使用保存的参数
    pca = PCA(n_components=1)
    pca.mean_ = mean  # 设置保存的成分
    pca.components_ = components  # 设置保存的成分
    pca.explained_variance_ = explained_variance  # 设置保存的成分
    
    # 使用保存的PCA参数降维新的数据
    reduced_data_2 = pca.transform(data_2_normalized)
    
    return reduced_data_2

@hydra.main(config_name='config_transition_flow1', config_path='configs')
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
    # assert torch.cuda.is_available()
    assert cfg.eval_episodes > 0, 'Must evaluate at least 1 episode.'
    cfg = parse_cfg(cfg)
    set_seed(cfg.seed)

    # Make environment
    env = make_env(cfg)

    # Load agent
    agent = TDMPC2_Flow(cfg)
    assert os.path.exists(cfg.checkpoint), f'Checkpoint {cfg.checkpoint} not found! Must be a valid filepath.'
    agent.load(cfg.checkpoint)
    print(colored(f'Checkpoint: {cfg.checkpoint}', 'blue', attrs=['bold']))
    
    # Evaluate
    print(colored(f'Evaluating agent on {cfg.task}, {cfg.sub_task}:', 'yellow', attrs=['bold']))

    fp = '/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/chenxinyan-240108120066/chenxinyan/tdmpc2/visual/visual_data_1296007625.pt'
    with open(fp, 'rb') as f:
        td = pickle.load(f)
    data_s = np.asarray(td['s'])
    data_next_s = np.asarray(td['next_s'])
    data_a = np.asarray(td['a'])

    predicted_next_x = np.zeros(data_next_s.shape)
    obs = data_s / 100.0 * 2 - 1  # norm -> [-1, 1]
    obs[:, -1] = (obs[:, -1] + 1) / 7.0 - 1
    for k in range(obs.shape[0]):
        s0 = torch.Tensor(obs[k]).unsqueeze(0).to(agent.device)
        action = torch.Tensor(data_a[k]).unsqueeze(0).to(agent.device)
        s = agent.model.next(s0, action, t_step=cfg.consistency_t_step)
        predicted_next_x[k] = s.cpu().detach().numpy()
    predicted_next_x = np.stack(predicted_next_x, axis=0)
    predicted_next_x[:, -1] = (predicted_next_x[:, -1] + 1) * 7. - 1
    predicted_next_s = (predicted_next_x + 1) * 100 / 2.

    pca_params = np.load('/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/chenxinyan-240108120066/chenxinyan/tdmpc2/visual/gt_pca_parameters.npy',
                         allow_pickle=True).item()
    reduced_s = apply_pca_parameters(data_s[:, 1:], pca_params)
    reduced_next_s = apply_pca_parameters(predicted_next_s[:, 1:], pca_params)

    # transform data
    x = reduced_s.astype(np.float32).squeeze()
    y = reduced_next_s.astype(np.float32).squeeze()
    mmin, mmax = x.min(), x.max()
    x = (x - mmin) / (mmax - mmin)
    x = x * 4 - 2
    y = (y - mmin) / (mmax - mmin)
    y = y * 4 - 2
    from sklearn.metrics import mean_squared_error
    mse_result = mean_squared_error(x, y)
    print(f'(s, next_s) MSE: {mse_result}')

    # interp
    interp_n = 10
    interped_x = np.zeros((x.shape[0], interp_n))
    for k in range(x.shape[0]):
        interped_x[k] = np.linspace(x[k], y[k], interp_n)
    x = interped_x

    # plot data with color of value
    plt.figure(figsize=(10, 6))
    plt.ylim([-2, 2])

    # 绘制每条线
    for i in range(x.shape[0]):
        plt.plot(range(1, interp_n+1), x[i], color='blue', alpha=0.03, marker=None)  # 不显示数据点

    # 设置图例、标题和标签等（如果需要）
    plt.title(f'transition_flow_wm, mse: {mse_result}')
    plt.xlabel('timestep')
    plt.ylabel('Value')

    plt.savefig('/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/chenxinyan-240108120066/chenxinyan/tdmpc2/visual/transition_flow_wm.png')


if __name__ == '__main__':
    evaluate()
