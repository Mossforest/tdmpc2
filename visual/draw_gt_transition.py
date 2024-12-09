import multiprocessing as mp
import os
import signal
import sys
import pickle

import matplotlib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from easydict import EasyDict
from rich.progress import track
from sklearn.datasets import make_swiss_roll
from grl.utils import set_seed

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
torch.set_float32_matmul_precision('high')
from easydict import EasyDict
from matplotlib import animation
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

def plot2d(data):
    plt.scatter(data[:, 0], data[:, 1])
    plt.show()

def plot2d_trajectories_2(data, video_save_path, iteration):
    if not os.path.exists(video_save_path):
        os.makedirs(video_save_path)
    
    T, B, _ = data.shape
    cmap = plt.get_cmap('YlGn')  # Colormap from blue to red
    norm = Normalize(vmin=0, vmax=3*T-1)
    
    plt.figure(figsize=(6, 6))
    plt.xlim([-3, 3])
    plt.ylim([-3, 3])
    
    # Plot trajectories with color transition
    for t in range(T):
        color = cmap(norm(t))
        plt.scatter(data[t, :300, 0], data[t, :300, 1], s=0.2, alpha=0.01, c=[color])

    # Plot start points
    color = cmap(norm(20))
    plt.scatter(data[0, :300, 0], data[0, :300, 1], s=10, alpha=1, c="yellow", label="Prior Distribution")

    # Plot end points
    color = cmap(norm(T-1))
    plt.scatter(data[-1, :300, 0], data[-1, :300, 1], s=10, alpha=1, c=[color], label="Posterior Distribution")
    
    plt.legend(loc="lower left")
    plt.savefig(os.path.join(video_save_path, f"trajectories_{iteration}.png"))
    plt.clf()

def plot2d_trajectories(data, video_save_path, iteration):
    if not os.path.exists(video_save_path):
        os.makedirs(video_save_path)
    
    T, B, _ = data.shape
    cmap = plt.get_cmap('YlGn')  # Colormap from blue to red
    norm = Normalize(vmin=0, vmax=3*T-1)
    
    plt.figure(figsize=(6, 6))
    plt.xlim([-3, 3])
    plt.ylim([-3, 3])
    
    # Plot trajectories with color transition
    for t in range(T):
        color = cmap(norm(t))
        plt.scatter(data[t, :300, 0], data[t, :300, 1], s=0.2, alpha=0.01, c=[color])

    # Plot start points
    color = cmap(norm(20))
    plt.scatter(data[0, :300, 0], data[0, :300, 1], s=10, alpha=1, c="yellow", label="Prior Distribution")

    # Plot end points
    color = cmap(norm(T-1))
    plt.scatter(data[-1, :300, 0], data[-1, :300, 1], s=10, alpha=1, c="greenyellow", label="Posterior Distribution")
    
    plt.legend(loc="lower left")
    plt.savefig(os.path.join(video_save_path, f"trajectories_{iteration}.png"))
    plt.clf()

def render_video(data_list, video_save_path, iteration, fps=100, dpi=100):
    if not os.path.exists(video_save_path):
        os.makedirs(video_save_path)
    fig = plt.figure(figsize=(6, 6))
    plt.xlim([-3, 3])
    plt.ylim([-3, 3])
    ims = []
    colors = np.linspace(0, 1, len(data_list))

    for i, data in enumerate(data_list):
        # image alpha frm 0 to 1
        im = plt.scatter(data[:, 0], data[:, 1], s=1)
        ims.append([im])
    ani = animation.ArtistAnimation(fig, ims, interval=0.1, blit=True)
    ani.save(
        os.path.join(video_save_path, f"iteration_{iteration}.mp4"), fps=fps, dpi=dpi
    )
    # clean up
    plt.close(fig)
    plt.clf()


def save_checkpoint(model, iteration, path):
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(
        dict(
            model=model.state_dict(),
            iteration=iteration,
        ),
        f=os.path.join(path, f"checkpoint_{iteration}.pt"),
    )


def save_checkpoint(diffusion_model, diffusion_model_iteration, path):
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(
        dict(
            diffusion_model=diffusion_model.state_dict(),
            diffusion_model_iteration=diffusion_model_iteration,
        ),
        f=os.path.join(path, f"checkpoint_{diffusion_model_iteration}.pt"),
    )


def pca_to_1d(data, n_components=1):
    """
    将n维数据通过PCA降维到1维，并返回PCA参数。
    
    参数:
    data : numpy.ndarray
        待降维的数据，形状为 (n_samples, n_features)。
    n_components : int, 默认为1
        降维后的目标维度数。
    
    返回:
    pca_parameters : dict
        PCA模型的参数，包括mean、components_和explained_variance_。
    reduced_data : numpy.ndarray
        降维后的数据。
    """
    # 标准化数据
    scaler = StandardScaler()
    data_normalized = scaler.fit_transform(data)
    
    # 初始化PCA模型
    pca = PCA(n_components=n_components)
    
    # 拟合PCA模型并降维数据
    reduced_data = pca.fit_transform(data_normalized)
    
    # 保存PCA模型参数
    pca_parameters = {
        'mean': scaler.mean_,
        'components_': pca.components_,
        'explained_variance_': pca.explained_variance_
    }
    
    return pca_parameters, reduced_data

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


if __name__ == "__main__":
    # get data
    fp = '/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/chenxinyan-240108120066/chenxinyan/tdmpc2/visual/visual_data_1296007625.pt'
    with open(fp, 'rb') as f:
        td = pickle.load(f)
    data_s = np.asarray(td['s'])
    data_next_s = np.asarray(td['next_s'])
    data_a = np.asarray(td['a'])

    # pca the data
    # the current data_s is (n, traj_n, 6); need change to (n, traj_n, 5) -> (n, traj_n, 1); is there any method to ingore (n, traj_n) as sample_n only?
    # n, traj_n, _ = data_s.shape
    # reduced_s = data_s[:, :, 1:].reshape(-1, 5)
    # pca_parameters, reduced_s = pca_to_1d(reduced_s, n_components=1)
    # reduced_s = reduced_s.reshape(n, traj_n, 1)
    # np.save('gt_pca_parameters.npy', pca_parameters)
    pca_params = np.load('gt_pca_parameters.npy', allow_pickle=True).item()
    reduced_s = apply_pca_parameters(data_s[:, 1:], pca_params)
    reduced_next_s = apply_pca_parameters(data_next_s[:, 1:], pca_params)

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

    # # sample (s, s') pair
    # sample_pair_n = 1000
    # pairs = []
    # for _ in range(sample_pair_n):
    #     start_idx = np.random.randint(0, n)
    #     start_traj_idx = np.random.randint(0, traj_n - 1)  # 因为需要采样两个连续的元素，所以最大值是100-1
    #     pair = x[start_idx, start_traj_idx:start_traj_idx+2, :].squeeze()
    #     pairs.append(pair)
    # x = np.stack(pairs, axis=0)

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
    plt.title(f'transition_datademo_gt, mse: {mse_result}')
    plt.xlabel('timestep')
    plt.ylabel('Value')

    plt.savefig('./transition_datademo_gt1.png')

