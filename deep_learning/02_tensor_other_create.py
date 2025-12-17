"""
torch.arange(), torch.linspace() 创建线性张量
torch.random.initial_seed(), torch.random.manual_seed() 设置随机种子
torch.rand(), torch.randn() 创建随机浮点类型张量
torch.randint(low, high, size=()) 创建随机整数类型张量

torch.ones(), torch.ones_like() 创建全1张量
torch.zeros(), torch.zeros_like() 创建全0张量
torch.full(), torch.full_like() 创建指定值张量
"""

import torch


# 创建线性张量
def dm01():
    # 创建指定范围的线性张量 -> arange
    t1 = torch.arange(0, 10, step=2)  # 0, 2, 4, 6, 8
    print(f"t1: {t1}, type(t1): {type(t1)}")
    print('-' * 30)
    # 创建指定范围的线性张量 -> linspace
    t2 = torch.linspace(1, 9, steps=5)  # 1.0, 3.0, 5.0, 7.0, 9.0
    print(f"t2: {t2}, type(t2): {type(t2)}")
    print('-' * 30)


# 创建随机张量
def dm02():
    # TODO: 设置随机种子
    # torch.initial_seed()  # 默认随机种子为当前时间戳
    torch.manual_seed(42)  # 设置随机种子为42

    # TODO: 创建随机浮点类型张量 -> rand(0~1均匀分布)
    t1 = torch.rand(size=(2, 3))
    print(f"t1: {t1}, type(t1): {type(t1)}")
    print('-' * 30)

    # TODO: 创建随机浮点类型张量 -> randn(正态分布)
    t2 = torch.randn(size=(2, 3))
    print(f"t2: {t2}, type(t2): {type(t2)}")
    print('-' * 30)

    # TODO: 创建随机整数类型张量
    t3 = torch.randint(low=0, high=10, size=(2, 3))
    print(f"t3: {t3}, type(t3): {type(t3)}")
    print('-' * 30)


# 创建指定值张量
def dm03():
    # 创建全1张量
    t1 = torch.ones(size=(3, 5))
    print(f"t1: {t1}, type(t1): {type(t1)}")
    print('-' * 30)
    t2 = torch.ones_like(torch.tensor([[1,2,3],[4,5,6]]))
    print(f"t2: {t2}, type(t2): {type(t2)}")
    print('-' * 30)
    # 创建全0张量
    t3 = torch.zeros(size=(3, 5))
    print(f"t3: {t3}, type(t3): {type(t3)}")
    print('-' * 30)
    t4 = torch.zeros_like(torch.tensor([[1,2,3],[4,5,6]]))
    print(f"t4: {t4}, type(t4): {type(t4)}")
    print('-' * 30)
    # 创建指定值张量
    t5 = torch.full(size=(3, 5), fill_value=7)
    print(f"t5: {t5}, type(t5): {type(t5)}")
    print('-' * 30)
    t6 = torch.full_like(torch.tensor([[1,2,3],[4,5,6]]), fill_value=3.14)
    print(f"t6: {t6}, type(t6): {type(t6)}")
    print('-' * 30)


# 测试函数
if __name__ == "__main__":
    # dm01()
    # dm02()
    dm03()