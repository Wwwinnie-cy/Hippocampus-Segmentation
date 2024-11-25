# 设定设备
import torch
from matplotlib import pyplot as plt
from torch.optim import Adam
import torch.nn as nn
from threeDunet import UNet3D
from dataset import create_datasets
from utils import train_one_epoch, validate
from torch.utils.data import Dataset, DataLoader
from utils import DiceLoss
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化模型、损失函数和优化器
model = UNet3D(n_channels=1, n_classes=3).to(device)
loss_fn = DiceLoss(3)
optimizer = Adam(model.parameters(), lr=1e-4)
train_dataset, valid_dataset = create_datasets('./imagesTr')
train_loader = DataLoader(train_dataset, batch_size=40, shuffle=True, num_workers=40)
valid_loader = DataLoader(valid_dataset, batch_size=40, shuffle=False, num_workers=40, pin_memory=True)
# 用于存储结果的列表
train_losses, valid_losses = [], []
metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}

num_epochs = 200
best_valid_loss = float('inf')
for epoch in range(num_epochs):
    train_loss = train_one_epoch(
        model, train_loader, loss_fn, optimizer, device)
    valid_loss = validate(
        model, valid_loader, loss_fn, device)

    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    # metrics['accuracy'].append((train_acc, valid_acc))
    # metrics['precision'].append((train_prec, valid_prec))
    # metrics['recall'].append((train_rec, valid_rec))
    # metrics['f1'].append((train_f1, valid_f1))
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'best_model.pth')

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss}, Validation Loss: {valid_loss}")


plt.figure(figsize=(12, 5))

plt.plot(train_losses, label='Train Loss')
plt.plot(valid_losses, label='Validation Loss')
plt.title('Loss over epochs')
plt.legend()
# 保存图像到文件
plt.savefig('training_validation_metrics.png')
plt.close()  # 关闭图形，释放资源

import json

# 准备要保存的数据
data_to_save = {
    "train_losses": train_losses,
    "valid_losses": valid_losses,
    # "accuracy": [(float(acc[0]), float(acc[1])) for acc in metrics['accuracy']],
    # "precision": [(float(prec[0]), float(prec[1])) for prec in metrics['precision']],
    # "recall": [(float(rec[0]), float(rec[1])) for rec in metrics['recall']],
    # "f1_score": [(float(f1[0]), float(f1[1])) for f1 in metrics['f1']]
}

# 保存为 JSON 文件
with open('training_validation_metrics.json', 'w') as f:
    json.dump(data_to_save, f, indent=4)