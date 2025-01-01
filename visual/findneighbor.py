import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn import manifold
from sklearn.metrics import pairwise_distances

def plot_histograms(tensor, name):
    """
    Plot histograms for each [n, 1, 1] group in a tensor of shape [n, 6, 1].
    
    Parameters:
    tensor (numpy.ndarray): A numpy array with shape [n, 6, 1] containing the data.
    """

    # 确保输入是numpy数组
    tensor = np.asarray(tensor)
    
    # 遍历每个[n, 1, 1]组
    group_data = tensor
    plt.figure()
    plt.hist(group_data, bins=100, color='skyblue', edgecolor='black')
    plt.xlim(0, 0.2)
    plt.ylim(0, 2000)
    plt.title(name)
    
    # 保存图形
    plt.savefig(f'./datadist_{name}.png')


fp = '/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/chenxinyan-240108120066/chenxinyan/industrialbenchmark-master/industrial_benchmark_python/data/data_demo_eval.pkl'
# fp = '/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/chenxinyan-240108120066/chenxinyan/industrialbenchmark-master/industrial_benchmark_python/data/data_demo.pkl'
with open(fp, 'rb') as f:
    td = pickle.load(f)

s = td['s'][:10, :-1, :]
s = s.reshape((-1, s.shape[-1]))
nexts = td['s'][:10, 1:, :]
tensor_nexts = nexts.reshape((-1, nexts.shape[-1]))
a = td['a'][:10, 1:, :]
a = a.reshape((-1, a.shape[-1]))
tensor_sa = np.concatenate((s, a), axis=-1)

# 0. norm each dim
for dim_i in range(tensor_sa.shape[-1]):
    mmin, mmax = np.min(tensor_sa[:, dim_i]), np.max(tensor_sa[:, dim_i])
    # print(f'dim {dim_i}: min {mmin}, max {mmax}')
    if mmin == mmax:
        tensor_sa[:, dim_i] = tensor_sa[:, dim_i] / mmin * 0.5
    else:
        tensor_sa[:, dim_i] = (tensor_sa[:, dim_i] - mmin) / (mmax - mmin)
    # mmin, mmax = np.min(tensor_sa[:, dim_i]), np.max(tensor_sa[:, dim_i])
    # print(f'after, dim {dim_i}: min {mmin}, max {mmax}')

# 1. try p = [2, 3, 5, 9]
p=2
print(tensor_sa.shape)
dist = pairwise_distances(tensor_sa, metric='minkowski', p=p)
print(dist.shape)

plot_histograms(dist.flatten(), f'tensor_sa_p{p}')