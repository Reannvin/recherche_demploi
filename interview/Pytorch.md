# Pytorch

## 创建张量

**`torch.tensor(data)`: 从数据创建张量**

这个函数会根据提供的数据创建一个新的张量。数据可以是列表、数组等。

```python
import torch

data = [1, 2, 3, 4, 5]
tensor_data = torch.tensor(data)
print(tensor_data)
```

**`torch.zeros(size)`: 创建元素全为0的张量**

创建一个指定大小的张量，其中所有元素的值都为0。

```python
import torch

size = (2, 3)
zeros_tensor = torch.zeros(size)
print(zeros_tensor)
```

**`torch.ones(size)`: 创建元素全为1的张量**

创建一个指定大小的张量，其中所有元素的值都为1。

```python
import torch

size = (2, 3)
ones_tensor = torch.ones(size)
print(ones_tensor)
```

**`torch.empty(size)`: 创建未初始化的张量**

创建一个指定大小的未初始化张量，其值取决于内存的状态。

```python
import torch

size = (2, 3)
empty_tensor = torch.empty(size)
print(empty_tensor)
```

**`torch.randn(size)`: 创建服从标准正态分布的张量**

创建一个指定大小的张量，其中的元素值是从标准正态分布中随机抽取的。

```python
import torch

size = (2, 3)
randn_tensor = torch.randn(size)
print(randn_tensor)
```

**`torch.arange(start, end, step)`: 创建一个范围内的一维张量**

创建一个一维张量，其中的元素值从起始值到结束值，步长为给定的步长。

```python
import torch

start = 0
end = 5
step = 1
arange_tensor = torch.arange(start, end, step)
print(arange_tensor)
```

**`torch.linspace(start, end, steps)`: 创建一个在指定范围内均匀间隔的张量**

创建一个一维张量，其中的元素值在指定范围内均匀分布。

```python
import torch

start = 0
end = 5
steps = 5
linspace_tensor = torch.linspace(start, end, steps)
print(linspace_tensor)
```

## 张量属性

**`.dtype`: 获取张量的数据类型**

返回张量中元素的数据类型。

```python
import torch

tensor = torch.tensor([1, 2, 3])
print(tensor.dtype)
```

**`.shape`: 获取张量的形状**

返回一个元组，表示张量的形状。

```python
import torch

tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(tensor.shape)
```

**`.device`: 获取张量所在的设备**

返回一个字符串，表示张量所在的设备，如'cpu'或'cuda:0'。

```python
import torch

tensor = torch.tensor([1, 2, 3])
print(tensor.device)
```

## 张量索引、切片与拼接

**`tensor[index]`: 索引操作**

使用索引来访问张量中的元素。

```python
import torch

tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
element = tensor[0, 1]  # Accesses the element at row 0, column 1
print(element)
```

**`tensor[start:end]`: 切片操作**

使用切片来获取张量的子张量。

```python
import torch

tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
sub_tensor = tensor[:, 1:]  # Slices the tensor to get all rows and columns starting from the second column
print(sub_tensor)
```

**`torch.cat(tensors, dim)`: 在给定维度上连接张量**

沿着指定维度将多个张量连接在一起。

```python
import torch

tensor1 = torch.tensor([[1, 2], [3, 4]])
tensor2 = torch.tensor([[5, 6], [7, 8]])
concatenated_tensor = torch.cat((tensor1, tensor2), dim=0)  # Concatenates along the row dimension
print(concatenated_tensor)
```

**`torch.stack(tensors, dim)`: 在新维度上堆叠张量**

在一个新的维度上堆叠多个张量。

```python
import torch

tensor1 = torch.tensor([1, 2, 3])
tensor2 = torch.tensor([4, 5, 6])
stacked_tensor = torch.stack((tensor1, tensor2), dim=1)  # Stacks tensors along a new dimension
print(stacked_tensor)
```

## 张量变换

**`tensor.view(shape)`: 返回给定形状的张量视图**

返回一个具有指定形状的新张量，原始张量的形状必须与新形状兼容。

```python
import torch

tensor = torch.tensor([[1, 2], [3, 4]])
reshaped_tensor = tensor.view(1, 4)  # Reshapes the tensor to a 1x4 tensor
print(reshaped_tensor)
```

**`tensor.reshape(shape)`: 改变张量的形状**

返回一个具有指定形状的新张量，原始张量的元素数量必须与新形状一致。

```python
import torch

tensor = torch.tensor([[1, 2], [3, 4]])
reshaped_tensor = tensor.reshape(1, 4)  # Reshapes the tensor to a 1x4 tensor
print(reshaped_tensor)
```

**`tensor.transpose(dim0, dim1)`: 交换两个维度**

交换张量中两个维度的位置。

```python
import torch

tensor = torch.tensor([[1, 2], [3, 4]])
transposed_tensor = tensor.transpose(0, 1)  # Swaps the first and second dimensions
print(transposed_tensor)
```

**`tensor.permute(\*dims)`: 按照指定顺序排列张量的维度**

按照给定顺序重新排列张量的维度。

```python
import torch

tensor = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
permuted_tensor = tensor.permute(1, 0, 2)  # Permutes the dimensions to (1, 0, 2)
print(permuted_tensor)
```

**`tensor.squeeze()`: 删除所有长度为1的维度**

删除张量中所有长度为1的维度。

```python
import torch

tensor = torch.tensor([[[1, 2], [3, 4]]])
squeezed_tensor = tensor.squeeze()  # Removes the single-dimensional entries
print(squeezed_tensor)
```

**`tensor.unsqueeze(dim)`: 在指定位置增加一个维度**

在指定位置增加一个长度为1的新维度。

```python
import torch

tensor = torch.tensor([[1, 2], [3, 4]])
unsqueezed_tensor = tensor.unsqueeze(0)  # Adds a dimension at index 0
print(unsqueezed_tensor)
```

## 数学运算

**`torch.add(x, y)`: 加法**

对两个张量进行逐元素加法运算。

```python
import torch

x = torch.tensor([1, 2, 3])
y = torch.tensor([4, 5, 6])
result = torch.add(x, y)
print(result)
```

**`torch.sub(x, y)`: 减法**

对两个张量进行逐元素减法运算。

```python
import torch

x = torch.tensor([1, 2, 3])
y = torch.tensor([4, 5, 6])
result = torch.sub(x, y)
print(result)
```

**`torch.mul(x, y)`: 乘法**

对两个张量进行逐元素乘法运算。

```python
import torch

x = torch.tensor([1, 2, 3])
y = torch.tensor([4, 5, 6])
result = torch.mul(x, y)
print(result)
```

**`torch.div(x, y)`: 除法**

对两个张量进行逐元素除法运算。

```python
import torch

x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([4.0, 5.0, 6.0])
result = torch.div(x, y)
print(result)
```

**`torch.matmul(x, y)`: 矩阵乘法**

计算两个张量的矩阵乘法。

```python
import torch

x = torch.tensor([[1, 2], [3, 4]])
y = torch.tensor([[5, 6], [7, 8]])
result = torch.matmul(x, y)
print(result)
```

**`torch.pow(base, exponent)`: 幂运算**

计算张量的幂。

```python
import torch

base = torch.tensor([1, 2, 3])
exponent = 2
result = torch.pow(base, exponent)
print(result)
```

**`torch.exp(tensor)`: 指数运算**

计算张量中所有元素的指数。

```python
import torch

tensor = torch.tensor([1.0, 2.0, 3.0])
result = torch.exp(tensor)
print(result)
```

**`torch.sqrt(tensor)`: 开方运算**

计算张量中所有元素的平方根。

```python
import torch

tensor = torch.tensor([1.0, 4.0, 9.0])
result = torch.sqrt(tensor)
print(result)
```

## 汇总统计

**`torch.sum(input)`: 求和**

计算张量中所有元素的和。

```python
import torch

tensor = torch.tensor([[1, 2], [3, 4]])
result = torch.sum(tensor)
print(result)
```

**`torch.mean(input)`: 求平均值**

计算张量中所有元素的平均值。

```python
import torch

tensor = torch.tensor([[1, 2], [3, 4]], dtype=torch.float)
result = torch.mean(tensor)
print(result)
```

**`torch.max(input)`: 求最大值**

找出张量中所有元素的最大值。

```python
import torch

tensor = torch.tensor([[1, 2], [3, 4]])
result = torch.max(tensor)
print(result)
```

**`torch.min(input)`: 求最小值**

找出张量中所有元素的最小值。

```python
import torch

tensor = torch.tensor([[1, 2], [3, 4]])
result = torch.min(tensor)
print(result)
```

**`torch.std(input)`: 求标准差**

计算张量中所有元素的标准差。

```python
import torch

tensor = torch.tensor([[1, 2], [3, 4]], dtype=torch.float)
result = torch.std(tensor)
print(result)
```

**`torch.var(input)`: 求方差**

计算张量中所有元素的方差。

```python
import torch

tensor = torch.tensor([[1, 2], [3, 4]], dtype=torch.float)
result = torch.var(tensor)
print(result)
```

## 梯度相关

**`tensor.requires_grad_()`: 标记张量需要计算梯度**

标记张量以便在反向传播中计算梯度。

```python
import torch

tensor = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
```

**`tensor.grad`: 获取张量的梯度**

获取张量的梯度值，前提是该张量需要计算梯度。

```python
import torch

tensor = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
tensor.sum().backward()
print(tensor.grad)
```

**`tensor.backward()`: 计算梯度**

计算张量的梯度值，前提是该张量需要计算梯度。

```python
import torch

tensor = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
tensor.sum().backward()
```

## 数据管理

**`tensor.to(device)`: 将张量移动到指定的设备上（如GPU）**

将张量移动到指定的设备上，例如GPU。

```python
import torch

tensor = torch.tensor([1, 2, 3])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tensor = tensor.to(device)
print(tensor.device)
```

**`torch.save(obj, f)`: 保存对象到文件**

将对象保存到文件中。

```python
import torch

tensor = torch.tensor([1, 2, 3])
torch.save(tensor, 'tensor.pt')  # Save tensor to file
```

**`torch.load(f)`: 从文件加载对象**

从文件中加载对象。

```python
import torch

tensor = torch.load('tensor.pt')  # Load tensor from file
print(tensor)
```

## 其他操作基础操作

**`torch.nn.functional.relu(input)`: 应用ReLU激活函数**

对输入张量应用ReLU激活函数。

```python
import torch.nn.functional as F
import torch

input = torch.tensor([-1, 0, 1], dtype=torch.float)
output = F.relu(input)
print(output)
```

**`torch.nn.Conv2d(in_channels, out_channels, kernel_size)`: 创建二维卷积层**

创建一个二维卷积层。

```python
import torch.nn as nn
import torch

conv_layer = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3)
input = torch.randn(1, 3, 64, 64)
output = conv_layer(input)
print(output.shape)
```

**`torch.optim.SGD(params, lr)`: 使用SGD优化器**

使用随机梯度下降（SGD）优化器来优化模型参数。

```python
import torch.optim as optim
import torch

params = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
optimizer = optim.SGD([params], lr=0.1)
```

## 自动求导（Autograd）

**自动求导是 PyTorch 中一个重要的功能，能够自动计算张量的梯度。**

```python
import torch

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x ** 2
y.backward(torch.tensor([1.0, 1.0, 1.0]))  # 计算 y 对 x 的梯度
print(x.grad)  # 输出梯度值
```

## 神经网络模块（nn.Module）

**使用 nn.Module 类来定义神经网络模型，可以方便地管理和组织模型的结构。**

```python
import torch.nn as nn
import torch

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)

model = Net()
```

## 数据加载与处理（Data Loading and Processing）

**使用 DataLoader 和 Dataset 类来加载和处理数据集。**

```python
import torch
from torch.utils.data import DataLoader, Dataset

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

data = [1, 2, 3, 4, 5]
dataset = CustomDataset(data)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
```

## 损失函数（Loss Functions）

**使用损失函数来衡量模型输出与真实标签之间的差异。**

```python
import torch.nn as nn
import torch

criterion = nn.CrossEntropyLoss()
output = torch.tensor([[0.1, 0.2, 0.7], [0.3, 0.6, 0.1]])
target = torch.tensor([2, 1])
loss = criterion(output, target)
print(loss)
```

## 优化器（Optimizers）

**使用优化器来更新模型的参数，常见的优化器包括 SGD、Adam 等。**

```python
import torch.optim as optim
import torch

model = Net()
optimizer = optim.SGD(model.parameters(), lr=0.01)
```

## 模型训练与验证（Model Training and Validation）

**使用 PyTorch 来训练和验证神经网络模型。**

```python
import torch.nn as nn
import torch.optim as optim
import torch

model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

for epoch in range(10):
    # 训练模型
    for data in dataloader:
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    # 验证模型
    with torch.no_grad():
        # 计算准确率等指标
```

## 模型保存与加载（Model Saving and Loading）

**在训练完成后，将模型保存到文件中以便后续使用。**

```python
import torch

torch.save(model.state_dict(), 'model.pth')  # 保存模型参数
```

## GPU 加速（GPU Acceleration）

**利用 GPU 加速计算可以显著提高模型训练的速度。**

```python
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)  # 将模型移动到 GPU 上
```

## 模型调优（Model Tuning）

**使用交叉验证和超参数搜索来调优模型，以提高模型性能。**

```python
from sklearn.model_selection import GridSearchCV
import torch

parameters = {'lr': [0.01, 0.1, 1.0]}
model = Net()
optimizer = optim.SGD(model.parameters(), lr=0.01)
grid_search = GridSearchCV(optimizer, parameters)
```

## 迁移学习（Transfer Learning）

**迁移学习是一种常见的训练技巧，可以使用预训练的模型来加速模型的训练过程。**

```python
import torchvision.models as models
import torch

pretrained_model = models.resnet18(pretrained=True)
# 将预训练模型的参数冻结
for param in pretrained_model.parameters():
    param.requires_grad = False
```