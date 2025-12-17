#!/usr/bin/env python3
import os, time
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# =========================
# 超参数（每次运行前在这里修改）
# =========================
RUN_NAME      = "A4_single_run"   # 随便命名本次实验
DATA_ROOT     = "./CIFAR10"     # data root
OUT_DIR       = "./outputs_planA_single"   # output directory
EPOCHS        = 100     # epochs
BATCH_SIZE    = 128     # batch size
NUM_WORKERS   = 8     # number of workers
LR            = 0.1     # learning rate(0.1)
WEIGHT_DECAY  = 0.0     # weight decay

# =========================
# 工具函数
# =========================
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

# =========================
# 模型（留空，待学生补全）
# =========================
class SmallCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        # TODO: 按“32→64→128 + ReLU + 池化；GAP；FC(128→10)”实现网络结构
        # 参考作业文档的层次与尺寸要求
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.layer4 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, num_classes)
        )
        # raise NotImplementedError("TODO: implement SmallCNN.__init__()")

    def forward(self, x):
        # TODO: 对应 __init__ 中的层次实现前向传播
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x
        # raise NotImplementedError("TODO: implement SmallCNN.forward()")

# =========================
# 数据（提示：题目 4 在此处自行加入数据增强）
# =========================
def get_dataloaders(data_root: str, batch_size: int, num_workers: int = 8, download: bool = True):
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2023, 0.1994, 0.2010)

    # 题目 1/2/3：仅 Normalize（无数据增强）
    # 题目 4：请你在此处“自行”加入数据增强
    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_set = torchvision.datasets.CIFAR10(root=data_root, train=True,  download=download, transform=train_tf)
    test_set  = torchvision.datasets.CIFAR10(root=data_root, train=False, download=download, transform=test_tf)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                               num_workers=num_workers, pin_memory=True)
    test_loader  = torch.utils.data.DataLoader(test_set,  batch_size=batch_size, shuffle=False,
                                               num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader

# =========================
# 训练 / 评估
# =========================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * x.size(0)
        _, pred = logits.max(1)
        correct += pred.eq(y).sum().item()
        total += y.size(0)
    return loss_sum/total, correct/total

@torch.no_grad()
def eval_model(model, loader, criterion, device):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        loss_sum += loss.item() * x.size(0)
        _, pred = logits.max(1)
        correct += pred.eq(y).sum().item()
        total += y.size(0)
    return loss_sum/total, correct/total

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(OUT_DIR); ensure_dir(out_dir)

    # 模型（需先完成 SmallCNN 才能运行）
    model = SmallCNN(num_classes=10).to(device)

    # 仅使用 SGD
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    train_loader, test_loader = get_dataloaders(DATA_ROOT, BATCH_SIZE, NUM_WORKERS, download=True)

    per_epoch = []
    best = {"val_acc": 0.0, "epoch": -1}
    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = eval_model(model, test_loader, criterion, device)  # 简化：用 test 观测趋势
        per_epoch.append([epoch, tr_loss, tr_acc, val_loss, val_acc])
        if val_acc > best["val_acc"]:
            best = {"val_acc": val_acc, "epoch": epoch}
        print(f"[{RUN_NAME}] epoch {epoch:03d}/{EPOCHS} | "
              f"tr_loss {tr_loss:.4f} tr_acc {tr_acc:.4f} | "
              f"val_loss {val_loss:.4f} val_acc {val_acc:.4f}")
    t1 = time.time()

    # 保存 CSV
    import csv
    csv_path = out_dir / f"{RUN_NAME}_log.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch","train_loss","train_acc","val_loss","val_acc"])
        w.writerows(per_epoch)

    # 画图
    try:
        epochs = [r[0] for r in per_epoch]
        tr_loss = [r[1] for r in per_epoch]
        val_loss = [r[3] for r in per_epoch]
        tr_acc  = [r[2] for r in per_epoch]
        val_acc = [r[4] for r in per_epoch]

        plt.figure()
        plt.plot(epochs, tr_loss, label="train_loss")
        plt.plot(epochs, val_loss, label="val_loss")
        plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend()
        plt.title(RUN_NAME)
        plt.savefig(out_dir / f"{RUN_NAME}_loss.png", dpi=160)
        plt.close()

        plt.figure()
        plt.plot(epochs, tr_acc, label="train_acc")
        plt.plot(epochs, val_acc, label="val_acc")
        plt.xlabel("epoch"); plt.ylabel("acc"); plt.legend()
        plt.title(RUN_NAME)
        plt.savefig(out_dir / f"{RUN_NAME}_acc.png", dpi=160)
        plt.close()
    except Exception as e:
        print("Plotting failed:", e)

    # 汇总
    summary = {
        "run": RUN_NAME,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "best_epoch": best["epoch"],
        "best_val_acc": best["val_acc"],
        "wall_time_sec": round(t1 - t0, 2),
    }
    print("Summary:", summary)

if __name__ == "__main__":
    main()
