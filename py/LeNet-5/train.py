#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''--------------------------------------------------
@Project -> File   ：python-learn -> train
@Author ：Tecypus
@Date   ：2021/9/29 16:49
@Desc   ：
--------------------------------------------------'''
import torch
from torch import nn
from net import MyLeNet5
from torch.optim import lr_scheduler
from torchvision import datasets, transforms
import os

# 数据转化为tensor格式
data_tansforms = transforms.Compose([
	transforms.ToTensor()
])

# 加载训练数据集
train_dataset = datasets.MNIST(root='./data', train=True, transform=data_tansforms,download=True)

train_dataloader = torch.utils.data.DataLoader(datasets=train_dataset, batch_size=16, shuffle=True)




