#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''--------------------------------------------------
@Project -> File   ：python-learn -> model
@Author ：Tecypus
@Date   ：2021/10/9 20:33
@Desc   ：
使用BN：
1.bias置为False
2.BN层放在conv层和relu层中间
--------------------------------------------------'''
import torch
import torch.nn as nn
class BasicBlock(nn.Module):
	expansion = 1

	def __init__(self, in_channel, out_channel, stride=1, downsample=None):
		super(BasicBlock, self).__init__()
		self.Conv1 = nn.Conv2d(in_channel=in_channel, out_channels=out_channel,
							   kernel_size=3, stride=stride, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(out_channel)
		self.relu = nn.ReLU()
		self.Conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
							   kernel_size=3, stride=1, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(out_channel)
		self.downsample = downsample

	def forward(self, x):
		identity = x # 捷径分支上的输出值
		if self.downsample is not None:
			identity = self.downsample(x)

		out = self.Conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.Conv2(out)
		out = self.bn2(out)

		out += identity
		out = self.relu(out)

		return out


class Bottleneck(nn.Module):
	expansion = 4 # 第三层卷积核个数是第一二的四倍

	def __init__(self, in_channel, out_channel, stride=1, downsample=None):
		super(Bottleneck, self).__init__()


