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

train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)

# 加载测试数据集
test_dataset = datasets.MNIST(root='./data', train=False, transform=data_tansforms,download=True)

test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=16, shuffle=True)

# 如果有显卡，可以用GPU
device = "cuda" if torch.cuda.is_available() else 'cpu'

# 调用net里面定义的网络模型，将模型转到GPU上
model = MyLeNet5().to(device)

# 定义一个损失函数(交叉熵损失)
loss_fn = nn.CrossEntropyLoss()

# 定义优化器
optimizer = torch.optim.SGD(model.parameters, lr=1e-3, momentum=0.9)

# 学习率每隔10分钟，变为原来的0.1
lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)


#定义训练函数
def train(dataloader, model, loss_fn, optimizer):
	loss, current, n = 0.0, 0.0, 0

	for bath, (X,y) in enumerate(dataloader):  # batch批次，X为图片，y为标签
		# 前向传播
		X, y = X.to(device), y.to(device)
		output = model(X)  # X放入模型，得到输出
		cur_loss = loss_fn(output, y)  # 计算当前输出与真实标签之间的损失值
		_, pred = torch.max(output, axis=1)  # 得到最大的概率值的索引和预测值

		cur_acc = torch.sum(y == pred) / output.shape[0]  # 算这一批次的精确度。output.shape[0]为16批次。分母看是多少个1.y==pred为1，反之为0.总得精确度。

		# 反向传播
		optimizer.zero_grad()
		cur_loss.backward()
		optimizer.step()  # 梯度更新

		loss += cur_loss.item()
		current += cur_acc.item()
		n += 1

	print("train_loss:{}".format(str(loss / n)))
	print("train_acc:{}".format(str(current / n)))


def val(dataloader, model, loss_fn):  # 验证，不经过反向传播
	model.eval()
	loss, current, n = 0.0, 0.0, 0
	with torch.no_grad():
		for bath, (X, y) in enumerate(dataloader):  # batch批次，X为图片，y为标签
			# 前向传播
			X, y = X.to(device), y.to(device)
			output = model(X)  # X放入模型，得到输出
			cur_loss = loss_fn(output, y)  # 计算当前输出与真实标签之间的损失值
			_, pred = torch.max(output, axis=1)  # 得到最大的概率值的索引和预测值

			cur_acc = torch.sum(y == pred) / output.shape[0]  # 算这一批次的精确度。output.shape[0]为16批次。分母看是多少个1.y==pred为1，反之为0.总得精确度。

			loss += cur_loss.item()
			current += cur_acc.item()
			n += 1
		print("val_loss:{}".format(str(loss / n)))
		print("val_acc:{}".format(str(current / n)))

		return current / n


# 训练
epoch = 50
min_acc = 0
for t in range(epoch):
	print(f'epoch{t+1}\n----------------------')
	train(train_dataloader, model, loss_fn, optimizer)
	a = val(test_dataloader, model, loss_fn)

	# 保存最好的模型权重
	if a > min_acc:
		folder = 'save_model'
		if not os.path.exists(folder):
			os.mkdir(folder)
		min_acc = a
		print("save the best model")

		torch.save(model.state_dict(), 'save_model/best_model.pth')

print('Done')







