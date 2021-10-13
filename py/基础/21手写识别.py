#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''--------------------------------------------------
@Project -> File   ：pylearn -> 21手写识别
@Author ：Tecypus
@Date   ：2021/8/28 9:41
@Desc   ：
--------------------------------------------------'''
from torchvision import transforms
import numpy as np
import torchvision

data = np.random.randint(0,255,size=12)
img = data.reshape(2,2,3)
img = transforms.ToTensor()(img)
print(img)
print('*' * 70)
