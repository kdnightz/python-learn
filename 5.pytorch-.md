# 2.在GPU上执行程序

1. 自定义的参数和数据 需要转化为cuda支持的tensor
2. model需要转化为cuda支持的model
3. 执行的结果 需要和cpu的tensor计算的时候
   1. tensor.cpu()把cuda的tensor转化为CPU的tensor
   2. 

# 3. 常见的优化算法介绍

## 3.1 梯度下降算法 （batch gradient descent BGD）

​	每次迭代需要把所有样本都送入，这样的好处是每次迭代都顾忌了全部的样本，做的是全局最优化。

​	数据太大，就太慢了

## 3.2 随机梯度下降法（SGD）



## 3.3 小批量梯度下降（MBGD）

## 3.4 动量法

