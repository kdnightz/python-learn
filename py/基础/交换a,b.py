#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''--------------------------------------------------
@Project -> File   ：pylearn -> 交换a,b
@Author ：Tecypus
@Date   ：2021/8/15 10:33
@Desc   ：
--------------------------------------------------'''
def swap(a,b):

	return b,a

a,b=10,20
a,b=swap(a,b)
print(a,b)

print('*'*30)

import random
i = 0
while i < 5:
	i += 1
	num = random.randint(1, 60)
	print(num)
