import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from tqdm import tqdm
import nibabel as nib
# 导入自定义模块
from threeDunet import UNet3D
from dataset import create_datasets, create_test_dataset
import os
import nibabel as nib
from torch.utils.data import Dataset
import numpy as np
import torchio as tio

# 使用定义好的数据集
train_dataset, valid_dataset = create_datasets('./imagesTr')
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=2, shuffle=False)

# 创建测试数据集和数据加载器
test_dataset = create_test_dataset('./imagesTs', './labelsTs')
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)
# 设置计算设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 初始化并加载模型
model = UNet3D(n_channels=1, n_classes=3)
model.load_state_dict(torch.load('best_model.pth', map_location=torch.device('cpu')))
model.to(device)
model.eval()
def infer_and_save(model, loader, device, filename=False):
    all_labels = []
    all_preds = []
    original_preds = []
    with torch.no_grad():
        if not filename:
            for images, labels in tqdm(loader, desc="Inference"):
                images = images.to(device)
                labels = labels.to(device).squeeze(0).long()
                outputs = model(images)
                outputs = F.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                all_labels.extend(labels.flatten().cpu().numpy())
                all_preds.extend(predicted.flatten().cpu().numpy())
                original_preds.extend(predicted.cpu().numpy())
        else:
            for images, labels, filenames in tqdm(loader, desc="Inference"):
                images = images.to(device)
                labels = labels.to(device).squeeze(0).long()
                outputs = model(images)
                outputs = F.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                all_labels.extend(labels.flatten().cpu().numpy())
                all_preds.extend(predicted.flatten().cpu().numpy())
                original_preds.extend(predicted.cpu().numpy())
    return all_labels, all_preds, original_preds

all_labels, all_preds, original_preds = infer_and_save(model, valid_loader, device)
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average='macro')
recall = recall_score(all_labels, all_preds, average='macro')
f1 = f1_score(all_labels, all_preds, average='macro')

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

all_labels, all_preds, original_preds = infer_and_save(model, test_loader, device, True)
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average='macro')
recall = recall_score(all_labels, all_preds, average='macro')
f1 = f1_score(all_labels, all_preds, average='macro')

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

def infer_and_save_nii(model, loader, device, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    with torch.no_grad():
        for images, labels, filenames in tqdm(loader, desc="Inference"):
            images = images.to(device)
            outputs = model(images)
            outputs = torch.argmax(F.softmax(outputs, dim=1), dim=1)

            # 将预测结果保存为 NIfTI 文件，并使用一致的文件名
            for j in range(outputs.shape[0]):
                pred_img = outputs[j].cpu().numpy().astype(np.int32)  # 转换为 int32 类型
                pred_nifti = nib.Nifti1Image(pred_img, affine=np.eye(4))
                pred_filename = os.path.join(output_dir, filenames[j])
                nib.save(pred_nifti, pred_filename)

infer_and_save_nii(model, test_loader, device, './saved_nii_results_resize64')
