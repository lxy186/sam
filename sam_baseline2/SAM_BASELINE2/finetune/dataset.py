import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import warnings
import albumentations as A

# 在文件开头添加以下代码来忽略特定警告
warnings.filterwarnings("ignore", message="Error fetching version info")

class BUSIDataset(Dataset):
    def __init__(self, data_root, transform=None, split='train'):
        """
        BUSI 乳腺肿瘤超声图像数据集
        Args:
            data_root: 数据集根目录
            transform: 图像变换
            split: 'train', 'val', 或 'test'
        """
        self.data_root = Path(data_root)
        self.transform = transform
        self.split = split
        
        # 收集数据路径
        self.samples = []
        
        # 处理良性和恶性样本
        for category in ['benign', 'malignant']:
            cat_dir = self.data_root / category
            if not cat_dir.exists():
                continue
                
            image_files = [f for f in os.listdir(cat_dir) if not f.endswith('_mask.png') and f.endswith('.png')]
            
            # 简单的训练/验证/测试分割
            # 可按需调整分割比例
            n_files = len(image_files)
            if split == 'train':
                image_files = image_files[:int(n_files * 0.7)]
            elif split == 'val':
                image_files = image_files[int(n_files * 0.7):int(n_files * 0.85)]
            else:  # test
                image_files = image_files[int(n_files * 0.85):]
            
            for img_file in image_files:
                img_path = cat_dir / img_file
                mask_path = cat_dir / img_file.replace('.png', '_mask.png')
                
                if mask_path.exists():
                    self.samples.append((img_path, mask_path, category))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, mask_path, category = self.samples[idx]
        
        # 读取图像和掩膜
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        mask = (mask > 0).astype(np.uint8)  # 二值化掩膜
        
        # 应用变换
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        # 检查是否已经是张量
        if not isinstance(image, torch.Tensor):
            # 转为 PyTorch 张量
            image = torch.from_numpy(np.transpose(image, (2, 0, 1))).float() / 255.0
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask).long()
        
        # 标签: 0为良性, 1为恶性
        label = 1 if category == 'malignant' else 0
        
        return {
            'image': image,
            'mask': mask,
            'label': label,
            'image_path': str(img_path)
        }

def build_data_loader(data_root, batch_size=4, num_workers=4):
    """构建数据加载器"""
    from albumentations.pytorch import ToTensorV2

    transform = A.Compose([
        A.Resize(1024, 1024),  # 调整大小到 1024x1024
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        ToTensorV2()  # 确保转换为 PyTorch 张量
    ])
    
    train_dataset = BUSIDataset(data_root, transform=transform, split='train')
    val_dataset = BUSIDataset(data_root, transform=transform, split='val')
    test_dataset = BUSIDataset(data_root, transform=transform, split='test')
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader 