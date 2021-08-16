# pytorchAPI的使用

## 	1. nn.Module

1. `__init__`

2. forward:完成一次 向前计算的过程

3. 
   
   ```python
   from torch import nn
   class Lr(nn.Module):
       def __init__(self):
           super(Lr,self).__init__() #继承父类init的参数
           self.linear = nn.Linear(1,1) #内部定义好了 w和b
           self.fc1 = nn.Linear(1,1)
           xxx
           
       def forward(Self,x):
           out = self.linear(x)
           out = fc1(out)# 多次操作
           out = relu(out)
           return out
       
       
   # 实例化模型
   model = Lr()
   #传入数据，计算结果
   predict = model(x)
   
   ```
   
## 1.2优化器类

   1. 对参数进行更新，优化
   
   2. torch.optim
      1. torch.optim.SGD(参数，学习率)
      2. torch.optim.Adam(参数，学习率)
   
   3. 注意，
      1. 参数可以使用`model.parameters()`来获取，获取模型中所有 `requires_grad=True` 的参数
      2. 优化类的使用方法
         1. 实例化
         2. 所有参数的梯度，将其值置为0
         3. 反向传播计算梯度
         4. 更新参数值
      3. 示例：
      	```python
      	optimizer = optim.SGD(model.parameters(), lr=1e-3) #2.实例化
      	optimizer.zero_grad() #2.梯度置为0
      	loss.backward() #3.计算梯度
      	optimizer.step() #4.更新参数的值
      	```
      	



## 1.3损失函数

1. 均方误差 ： `nn.MSELoss()` 常用于分类问题
2. 交叉熵损失： `nn.CrossEntropyLoss()` ，常用于逻辑回归



## 1.4 API实现线性回归

```python
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
my_linear.eval()  # 设置模型为评估模式，即预测模式 == my_linear.tran(false) # model.train(mode=True)表示设置模型为训练模式
predict = my_linear(x)
predict = predict.data.numpy()
plt.scatter(x.data.numpy(), y_true.data.numpy(), c="r")
plt.plot(x.data.numpy(), predict)
plt.show()
```

