"""
张量 -> 存储同一类元素的容器，元素值必须为数值

张量的基本创建方式：
    torch.tensor  根据指定数据创建张量
    torch.Tensor  根据形状创建张量，也可以根据指定数据创建张量
    torch.IntTensor, torch.FloatTensor, torch.DoubleTensor  根据指定类型创建张量
"""

import torch
import numpy as np


# 1. torch.tensor 创建张量
def dm01():
    # 创建标量张量
    t1 = torch.tensor(5)
    print(f"t1: {t1}, type(t1): {type(t1)}")
    print('-' * 30)
    # 列表转换为张量
    data = [[1,2,3],[4,5,6]]
    t2 = torch.tensor(data, dtype=torch.float32)  # dtype指定数据类型
    print(f"t2: {t2}, type(t2): {type(t2)}")
    print('-' * 30)
    # numpy数组转换为张量
    data = np.random.randint(0, 10, size=(2,3))
    t3 = torch.tensor(data)
    print(f"t3: {t3}, type(t3): {type(t3)}")
    print('-' * 30)
    # 尝试直接创建指定维度的张量
    # t4 = torch.tensor(2, 3)  # 错误示范


# 2. torch.Tensor 创建张量
def dm02():
    # 创建标量张量
    t1 = torch.Tensor(5)
    print(f"t1: {t1}, type(t1): {type(t1)}")
    print('-' * 30)
    # 列表转换为张量
    data = [[1,2,3],[4,5,6]]
    t2 = torch.Tensor(data)
    print(f"t2: {t2}, type(t2): {type(t2)}")
    print('-' * 30)
    # numpy数组转换为张量
    data = np.random.randint(0, 10, size=(2,3))
    t3 = torch.Tensor(data)
    print(f"t3: {t3}, type(t3): {type(t3)}")
    print('-' * 30)
    # 直接创建指定维度的张量
    t4 = torch.Tensor(2, 3)
    print(f"t4: {t4}, type(t4): {type(t4)}")
    print('-' * 30)


# 3. torch.IntTensor, torch.FloatTensor, torch.DoubleTensor 创建张量
def dm03():
    # 创建标量张量
    t1 = torch.IntTensor(5)
    print(f"t1: {t1}, type(t1): {type(t1)}")
    print('-' * 30)
    # 列表转换为张量
    data = [[1,2,3],[4,5,6]]
    t2 = torch.IntTensor(data)
    print(f"t2: {t2}, type(t2): {type(t2)}")
    print('-' * 30)
    # numpy数组转换为张量
    data = np.random.randint(0, 10, size=(2,3))
    t3 = torch.IntTensor(data)
    print(f"t3: {t3}, type(t3): {type(t3)}")
    print('-' * 30)
    # 如果类型不匹配，会尝试自动转换
    t4 = torch.FloatTensor(data)
    print(f"t4: {t4}, type(t4): {type(t4)}")
    print('-' * 30)


# 测试函数
if __name__ == "__main__":
    # dm01()
    # dm02()
    dm03()