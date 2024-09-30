

import torch.nn as nn

model = nn.Sequential(
    nn.Conv2d(1, 20, 5),  # 第一个卷积层，输入通道1，输出通道20，卷积核大小5
    nn.ReLU(),            # ReLU激活函数
    nn.Conv2d(20, 64, 5), # 第二个卷积层，输入通道20，输出通道64，卷积核大小5
    nn.ReLU(),            # ReLU激活函数
    nn.Flatten(),         # 将多维输入一维化
    nn.Linear(64 * 5 * 5, 1024),  # 全连接层，输入特征64*5*5，输出特征1024
    nn.ReLU(),            # ReLU激活函数
    nn.Linear(1024, 10)   # 最后一个全连接层，输出特征10
)

# 打印模型结构
print(model)