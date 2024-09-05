import torch
from tensordict import TensorDict

# 读取.pt文件
model_path = '/mnt/nfs/chenxinyan/tdmpc2/data/single-test/mw-bin-picking_0.1.pt'
model = torch.load(model_path)

# 创建一个新的空TensorDict
new_tensordict = TensorDict()

# 遍历模型权重并创建相同名称的空张量
for key, value in model.items():
    # 假设所有权重都是相同数据类型和形状的张量
    new_tensordict[key] = torch.zeros_like(value)
    print(f"['{key}': {value.shape}]")

print(model['action'][0][0])
print(new_tensordict)