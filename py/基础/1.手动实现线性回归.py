#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''--------------------------------------------
@Project -> File   ：pylearn -> 1.手动实现线性回归
@Author ：Tecypus
@Date   ：2021/8/10 16:16
@Desc   ：
--------------------------------------------'''

import torch
from torch import nn  # import torch.nn as nn
from torch import optim # import torch.optim.SGD
import numpy as np
from matplotlib import pyplot as plt

# 1.定义数据
x = torch.rand([50, 1])
y_true = x * 3 + 0.8


# 2.定义模型
class MyLinear(nn.Module):
	def __init__(self):
		# 继承父类init
		super(MyLinear,self).__init__()
		self.linear = nn.Linear(1, 1)

	def forward(self, x):
		out = self.linear(x)
		return out


# 2.实例化模型，loss，和优化器
my_linear = MyLinear()
loss_fn = nn.MSELoss()
optimizer = optim.SGD(my_linear.parameters(), lr=1e-3)

# 3.训练模型
for i in range(3000):
	y_predict = my_linear(x)  # 3.1获取预测值
	loss = loss_fn(y_true, y_predict)  # 3.2计算损失
	optimizer.zero_grad()  # 3.3梯度归零
	loss.backward()  # 3.4计算梯度 反向传播
	optimizer.step()  # 3.5更新梯度
	if (i + 1) % 20 == 0:
		print('Epoch[{}/{}],loss:{:.6f}'.format(i, 30000, loss.data))
		# params = list(my_linear.parameters())
		# print(loss.item(),params[0].item(),params[1].item())

# 4.模型评估
my_linear.eval()  # 设置模型为评估模式，即预测模式 == my_linear.tran(false)
predict = my_linear(x)
predict = predict.data.numpy()
plt.scatter(x.data.numpy(), y_true.data.numpy(), c="r")
plt.plot(x.data.numpy(), predict)
plt.show()