#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''--------------------------------------------------
@Project -> File   ：python-learn -> test
@Author ：Tecypus
@Date   ：2021/10/4 9:41
@Desc   ：
--------------------------------------------------'''
import torch
from net import MyLeNet5
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision.transforms import ToPILImage


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

model.state_dict(torch.load("c:/a/b.pth"))

# 获取结果
classes = [
	'0',
	'1',
	'2',
	'3',
	'4',
	'5',
	'6',
	'7',
	'8',
	'9'
]

# 把 Tensor 转化为 图片，方便可视化
show = ToPILImage()

# 进入验证
for i in range(20):
	X, y = test_dataloader[i][0], test_dataloader[i][1]
	show(X).show()

	X = Variable(torch.unsqueeze(X, dim=0).float(), requires_grad=False).to(device)
	with torch.no_grad():  # 张量没有梯度的情况下
		pred = model(X)

		predicted, actual = classes[torch.argmax(pred[0])], classes[y]

		print(f'predicted:"{predicted}", actual: "{actual}"')





