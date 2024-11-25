import os
import torch
import nibabel as nib
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torchio as tio

# 定义一个数据增强变换
target_shape = (64, 64, 64)  # 定义目标形状

transform = tio.Compose([
    tio.RescaleIntensity(out_min_max=(0, 1)),  # 将图像强度重新缩放到 [0, 1]
    tio.Resize(target_shape),  # 调整图像大小
    tio.RandomAffine(  # 随机仿射变换
        degrees=10,  # 可以指定单个值或范围
        scales=(0.9, 1.1),
        isotropic=False,
        center='image',
        translation=(-10, 10),  # 正确的关键字应该是 'translation'
    ),
    tio.RandomFlip(axes=(0,)),  # 沿Z轴翻转
    tio.RandomNoise(p=0.5),  # 以50%的概率添加随机噪声
    tio.RandomBiasField(p=0.5),  # 以50%的概率添加随机强度非均匀性
])

# 定义验证集和测试集的变换（仅调整大小和重新缩放强度）
val_test_transform = tio.Compose([
    tio.RescaleIntensity(out_min_max=(0, 1)),  # 将图像强度重新缩放到 [0, 1]
    tio.Resize(target_shape),  # 调整图像大小
])

# 将这个变换应用到你的Dataset类
class NiiDataset(Dataset):
    def __init__(self, image_dir, mask_dir, files, transform=None, return_filenames=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.files = files
        self.transform = transform
        self.return_filenames = return_filenames

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_file = self.files[idx]
        img_path = os.path.join(self.image_dir, img_file)
        mask_path = os.path.join(self.mask_dir, img_file)
        
        img_nii = nib.load(img_path)
        img = img_nii.get_fdata()
        mask_nii = nib.load(mask_path)
        mask = mask_nii.get_fdata()

        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)  # 添加一个通道维度
        mask = torch.tensor(mask).unsqueeze(0)

        subject = tio.Subject(
            image=tio.ScalarImage(tensor=img),
            mask=tio.LabelMap(tensor=mask)
        )

        if self.transform:
            subject = self.transform(subject)

        if self.return_filenames:
            return subject['image'][tio.DATA], subject['mask'][tio.DATA], img_file
        else:
            return subject['image'][tio.DATA], subject['mask'][tio.DATA]

def create_datasets(directory, test_size=0.2, random_state=42):
    # 列出所有nii图像文件
    all_files = [f for f in sorted(os.listdir(directory)) if f.endswith('.nii.gz')]
    # 划分训练集和验证集
    train_files, test_files = train_test_split(all_files, test_size=test_size, random_state=random_state)
    train_dataset = NiiDataset(directory, directory.replace('imagesTr', 'labelsTr'), train_files, transform=transform)
    test_dataset = NiiDataset(directory, directory.replace('imagesTr', 'labelsTr'), test_files, transform=val_test_transform)

    return train_dataset, test_dataset

def create_test_dataset(image_dir, mask_dir):
    # 列出所有nii图像文件
    test_files = [f for f in sorted(os.listdir(image_dir)) if f.endswith('.nii.gz')]
    test_dataset = NiiDataset(image_dir, mask_dir, test_files, transform=val_test_transform, return_filenames=True)
    return test_dataset

# 使用定义好的数据集
train_dataset, valid_dataset = create_datasets('./imagesTr')
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=2, shuffle=False)

# 创建测试数据集和数据加载器
test_dataset = create_test_dataset('./imagesTs', './labelsTs')
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

if __name__ == '__main__':
    # 检查数据
    for images, masks in train_loader:
        print("Training batch - images:", images.shape, "masks:", masks.shape)
    for images, masks in valid_loader:
        print("Validation batch - images:", images.shape, "masks:", masks.shape)
    for images, masks, filenames in test_loader:
        print("Test batch - images:", images.shape, "masks:", masks.shape, "filenames:", filenames)
