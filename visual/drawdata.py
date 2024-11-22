import matplotlib.pyplot as plt
import numpy as np
import pickle

def plot_histograms(tensor, name):
    """
    Plot histograms for each [n, 1, 1] group in a tensor of shape [n, 6, 1].
    
    Parameters:
    tensor (numpy.ndarray): A numpy array with shape [n, 6, 1] containing the data.
    """

    # 确保输入是numpy数组
    tensor = np.asarray(tensor)
    feature_dim = tensor.shape[1]
    
    # 创建一个图形和子图
    fig, axs = plt.subplots(1, feature_dim, figsize=(15, 5))  # 1行6列的子图
    
    # 遍历每个[n, 1, 1]组
    for i in range(feature_dim):
        # 获取第i组数据
        group_data = tensor[:, i]
        
        # 在对应的子图上绘制直方图
        try:
            axs[i].hist(group_data, bins=20, color='skyblue', edgecolor='black')
            axs[i].set_title(f'Group {i+1}')
        except TypeError:
            axs.hist(group_data, bins=20, color='skyblue', edgecolor='black')

    plt.suptitle(f'Histograms of {name}')
    # 调整子图间距
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # 调整subplots的布局，留出空间给总标题
    
    # 保存图形
    plt.savefig(f'./datadist_50_{name}.png')
    

# 假设你有一个numpy数组tensor，你可以这样调用这个函数：
# tensor = np.random.randn(100, 6, 1)  # 随机生成一个[n, 6, 1]的数组作为示例
# plot_histograms(tensor)

fp = '/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/chenxinyan-240108120066/chenxinyan/industrialbenchmark-master/industrial_benchmark_python/data/data_50_eval.pkl'
# fp = '/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/chenxinyan-240108120066/chenxinyan/industrialbenchmark-master/industrial_benchmark_python/data/data_demo.pkl'
with open(fp, 'rb') as f:
    td = pickle.load(f)

for k, v in td.items():
    if v.ndim != 3:
        assert v.ndim == 2
        v = np.expand_dims(v, axis=2)
        print('tensor shape:', v.shape)
    v = v.reshape(-1, v.shape[-1])
    plot_histograms(v, k)
    # print(k, v.shape)
