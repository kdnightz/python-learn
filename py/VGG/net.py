#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''--------------------------------------------------
@Project -> File   ：python-learn -> net
@Author ：Tecypus
@Date   ：2021/10/4 10:56
@Desc   ：1.堆叠多个3*的卷积核来替代大尺度卷积核
--------------------------------------------------'''
# conv 3*3,s=1,p=1    maxpool 2*2 s=2
import torch
from torch import nn


# 定义网络模型
class VGG(nn.Module):
	'''
	1.input 224*224*3
	2. 两次3*3conv ——> 224*224*64
	3. maxpool1 -> 112*112*64
	4. 两次conv -> 112*112*128
	5. maxpool2 -> 56*56*128
	6. 三次conv -> 56*56*256
	7. maxpool3 -> 28*28*256
	8. 三次conv -> 28*28*512
	9. maxpool4 -> 14*14*512
	10. 三次conv -> 14*14*512
	11. maxpool5 -> 7*7*512
	12. 全连接1-4096
	13. 全连接2-4096
	14. 全连接3-1000  这里不再relu
	15. softmax激活函数
	'''
	def __init__(self, features, class_num=1000, init_weight=False):
		super(VGG, self).__init__()
		self.features = features
		self.classifier = nn.Sequential(
			nn.Dropout(p=0.5),  # 随机失活，防止过拟合
			nn.Linear(512*7*7, 2048),
			nn.ReLU(True),
			nn.Dropout(p=0.5),
			nn.Linear(2048, 2048),
			nn.ReLU(True),
			nn.Linear(2048, class_num)
		)
		if init_weight:
			self._initialize_weights()

	def forward(self, x):
		x = self.features(x)
		x = torch.flatten(x, start_dim=1)
		x = self.classifier(x)
		return x

	def _initialize_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.xavier_uniform_(m.weight)
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)
			elif isinstance((m, nn.Linear)):
				nn.init.xavier_uniform_(m.weight)
				nn.init.constant_(m.bias, 0)


def make_features(cfg:list):
	layers = []
	in_channels = 3
	for v in cfg:
		if v == 'M':  # 最大池化层
			layers += [nn.MaxPool2d(in_channels, v, kernel_size=2, stride=2)]
		else:
			conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
			layers += [conv2d, nn.ReLU(True)]
			in_channels = v
	return nn.Sequential(*layers)  # 非关键字参数


cfgs = {
	'vgg16' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
}


def vgg(model_name="vgg16", **kwargs):
	try:
		cfg = cfgs[model_name]
	except:
		print(f'Warning:model num {model_name} is not in cfgs dict!')
		assert ''
		exit(-1)
	model = VGG(make_features(cfg), **kwargs)
	return model


if __name__ == "__main__":
	vgg()


