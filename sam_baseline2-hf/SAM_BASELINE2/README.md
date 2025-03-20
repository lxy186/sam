# SAM 模型微调项目说明文档

本项目基于 Meta 的 Segment Anything Model (SAM) 进行微调，用于乳腺超声图像分割任务。本文档详细介绍项目的结构、运行方法和实现细节。

## 目录
1. [项目简介](#项目简介)
2. [环境配置](#环境配置)
3. [项目结构](#项目结构)
4. [数据准备](#数据准备)
5. [运行说明](#运行说明)
6. [实现细节](#实现细节)
7. [参数说明](#参数说明)
8. [训练过程](#训练过程)
9. [评估方法](#评估方法)
10. [常见问题](#常见问题)

## 项目简介

本项目在 SAM 模型的基础上进行微调，主要用于乳腺超声图像的分割任务。SAM 是一个强大的分割基础模型，但直接应用于医学图像可能效果不佳。通过微调，我们可以让模型更好地适应特定的医学图像分割任务。

### 为什么选择 SAM？
- 强大的通用分割能力
- 支持提示点引导的交互式分割
- 预训练模型具有良好的特征提取能力
- 模块化设计便于微调

### 主要特点
- 基于 SAM 模型进行微调
- 使用 BUSI 数据集进行训练
- 支持多种评估指标（Dice、HD95、特异性）
- 提供详细的可视化功能
- 支持断点续训和模型保存

### 创新点
1. 采用中心点提示策略
2. 冻结图像编码器以保留通用特征
3. 使用组合损失函数优化分割效果
4. 实现逐图像处理避免批处理问题

## 环境配置

### 1. 基本环境要求
```bash
Python >= 3.8
CUDA >= 11.3 (如果使用GPU)
PyTorch >= 1.12.0
```

### 2. 安装依赖
```bash
# 基础依赖
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu113
pip install opencv-python matplotlib tqdm tensorboard scipy

# 数据增强库
pip install albumentations

# 可选：用于可视化
pip install jupyter notebook
```

### 3. 安装 SAM
```bash
git clone https://github.com/facebookresearch/segment-anything.git
cd segment-anything
pip install -e .
```

### 4. 验证安装
```python
import torch
import segment_anything
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"SAM version: {segment_anything.__version__}")
```

## 项目结构
```
SAM_BASELINE2/
├── finetune/
│   ├── dataset.py     # 数据集加载和预处理
│   ├── train.py       # 训练脚本
│   ├── evaluate.py    # 评估脚本
│   └── utils.py       # 工具函数（损失计算、评估指标等）
├── checkpoint/        # 预训练模型存放
│   └── sam_vit_h_4b8939.pth  # SAM预训练权重
├── output/           # 输出目录
│   ├── best_model.pth        # 最佳模型权重
│   ├── checkpoints/          # 训练检查点
│   └── tensorboard/          # 训练日志
└── Dataset_BUSI_with_GT/     # 数据集目录
```

## 数据准备

### 1. 下载数据集
从 [BUSI 数据集官方页面](https://scholar.cu.edu.eg/?q=afahmy/pages/dataset) 下载数据集。

### 2. 数据集结构
```
Dataset_BUSI_with_GT/
├── benign/                # 良性肿瘤图像
│   ├── benign (1).png    # 原始图像
│   ├── benign (1)_mask.png   # 对应掩码
│   └── ...
└── malignant/            # 恶性肿瘤图像
    ├── malignant (1).png
    ├── malignant (1)_mask.png
    └── ...
```

### 3. 数据预处理
- 图像统一调整为 1024x1024 分辨率
- 掩码二值化处理
- 数据集按 7:1.5:1.5 比例划分训练/验证/测试集

## 运行说明

### 1. 准备工作
1. 下载 SAM 预训练模型
```bash
# 下载 ViT-H 预训练权重
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
mv sam_vit_h_4b8939.pth checkpoint/
```

2. 创建必要目录
```bash
mkdir -p checkpoint output/checkpoints output/tensorboard
```

### 2. 训练模型
```bash
# 基础训练
python finetune/train.py \
    --batch_size 2 \
    --epochs 20 \
    --data_path Dataset_BUSI_with_GT \
    --checkpoint checkpoint/sam_vit_h_4b8939.pth \
    --output_dir output \
    --lr 1e-5

# 使用较小模型训练（显存不足时）
python finetune/train.py \
    --model_type vit_b \
    --batch_size 4 \
    --epochs 20 \
    --data_path Dataset_BUSI_with_GT \
    --checkpoint checkpoint/sam_vit_b.pth \
    --output_dir output \
    --lr 1e-5
```

### 3. 评估模型
```bash
# 基础评估
python finetune/evaluate.py \
    --data_path Dataset_BUSI_with_GT \
    --checkpoint output/best_model.pth \
    --output_dir evaluation

# 生成可视化结果
python finetune/evaluate.py \
    --data_path Dataset_BUSI_with_GT \
    --checkpoint output/best_model.pth \
    --output_dir evaluation \
    --visualize \
    --num_vis_samples 10  # 可视化样本数量
```

### 4. 训练监控
```bash
# 启动TensorBoard
tensorboard --logdir output/tensorboard
```

## 实现细节

### 1. 模型架构
SAM 模型包含三个主要组件：
1. 图像编码器 (Image Encoder)
   - 使用 ViT 架构
   - 提取图像特征
   - 在微调时参数冻结
   
2. 提示编码器 (Prompt Encoder)
   - 处理点、框等提示信息
   - 生成提示嵌入
   - 微调时更新参数
   
3. 掩码解码器 (Mask Decoder)
   - 生成分割掩码
   - 使用 Transformer 架构
   - 微调时更新参数

### 2. 微调策略
1. 参数冻结
```python
# 冻结图像编码器
for param in sam.image_encoder.parameters():
    param.requires_grad = False
```

2. 提示点生成
```python
# 使用图像中心点作为提示
h, w = image.shape[-2:]
point = torch.tensor([[[w//2, h//2]]], device=device)
point_label = torch.ones(1, 1, device=device)
```

3. 损失函数设计
```python
# Dice Loss
def dice_loss(pred, target):
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    return 1 - (2.0 * intersection) / (union + 1e-6)

# 组合损失
loss = 0.5 * dice_loss(pred, target) + 0.5 * bce_loss(pred, target)
```

### 3. 数据增强策略
详细的数据增强配置：
```python
transform = A.Compose([
    A.Resize(1024, 1024),  # 调整大小
    A.HorizontalFlip(p=0.5),  # 水平翻转
    A.VerticalFlip(p=0.5),   # 垂直翻转
    A.RandomRotate90(p=0.5), # 随机90度旋转
    A.RandomBrightnessContrast(
        brightness_limit=0.2,
        contrast_limit=0.2,
        p=0.2
    ),  # 亮度对比度调整
    A.GaussianBlur(p=0.2),  # 高斯模糊
    A.GaussNoise(p=0.2),    # 高斯噪声
])
```

## 参数说明

### 1. 训练参数
```python
# 基础参数
--batch_size: 批次大小（默认：2）
--epochs: 训练轮数（默认：20）
--lr: 学习率（默认：1e-5）
--weight_decay: 权重衰减（默认：1e-4）

# 路径参数
--data_path: 数据集路径
--checkpoint: 预训练模型路径
--output_dir: 输出目录

# 硬件参数
--device: 使用设备（cuda/cpu）
--num_workers: 数据加载线程数（默认：4）

# 模型参数
--model_type: SAM模型类型（vit_h/vit_l/vit_b）
--seed: 随机种子（默认：42）
```

### 2. 评估参数
```python
# 基础参数
--batch_size: 批次大小（默认：4）
--data_path: 数据集路径
--checkpoint: 模型检查点路径
--output_dir: 输出目录

# 可视化参数
--visualize: 是否生成可视化结果
--num_vis_samples: 可视化样本数量
--save_masks: 是否保存预测掩码
```

## 训练过程

### 1. 训练流程详解
1. 数据准备阶段
   - 加载数据集
   - 应用数据增强
   - 批次组织

2. 模型前向传播
   ```python
   # 图像编码
   image_embedding = sam.image_encoder(image)
   
   # 提示点生成和编码
   sparse_embedding, dense_embedding = sam.prompt_encoder(
       points=(point, point_label)
   )
   
   # 掩码预测
   mask_prediction, _ = sam.mask_decoder(
       image_embeddings=image_embedding,
       sparse_prompt_embeddings=sparse_embedding,
       dense_prompt_embeddings=dense_embedding
   )
   ```

3. 损失计算和反向传播
   ```python
   # 损失计算
   loss_dice = dice_loss(pred_mask, target_mask)
   loss_bce = bce_loss(pred_mask, target_mask)
   loss = 0.5 * loss_dice + 0.5 * loss_bce
   
   # 反向传播
   optimizer.zero_grad()
   loss.backward()
   optimizer.step()
   ```

4. 验证评估
   - 计算验证集指标
   - 更新最佳模型
   - 记录训练日志

### 2. 训练监控
1. 终端输出
   ```
   Epoch [1/20] Training: Loss: 0.4521 | Dice: 0.7234 | LR: 0.00001
   Epoch [1/20] Validation: Loss: 0.3890 | Dice: 0.7456 | HD95: 12.345 | Spec: 0.9234
   ```

2. TensorBoard 指标
   - 损失曲线
   - Dice系数变化
   - HD95变化
   - 特异性变化
   - 学习率变化

3. 检查点保存
   ```python
   # 保存最佳模型
   torch.save({
       'epoch': epoch,
       'model_state_dict': model.state_dict(),
       'optimizer_state_dict': optimizer.state_dict(),
       'best_dice': best_dice,
   }, 'output/best_model.pth')
   ```

## 评估方法

### 1. 定量评估
1. Dice系数
   ```python
   def calculate_dice(pred, target):
       intersection = (pred * target).sum()
       union = pred.sum() + target.sum()
       return 2.0 * intersection / (union + 1e-6)
   ```

2. HD95 (95%豪斯多夫距离)
   ```python
   def compute_hd95(pred, target):
       pred_points = np.argwhere(pred)
       target_points = np.argwhere(target)
       d1, _, _ = directed_hausdorff(pred_points, target_points)
       d2, _, _ = directed_hausdorff(target_points, pred_points)
       return max(d1, d2) * 0.95
   ```

3. 特异性
   ```python
   def calculate_specificity(pred, target):
       TN = np.sum((pred == 0) & (target == 0))
       FP = np.sum((pred == 1) & (target == 0))
       return TN / (TN + FP + 1e-6)
   ```

### 2. 可视化评估
1. 结果展示
   - 原始超声图像
   - 真实分割掩码
   - 预测分割掩码
   - 掩码叠加效果

2. 保存格式
   ```python
   plt.figure(figsize=(15, 5))
   plt.subplot(131)
   plt.imshow(image)
   plt.title('Original Image')
   
   plt.subplot(132)
   plt.imshow(image)
   plt.imshow(gt_mask, alpha=0.5, cmap='Reds')
   plt.title('Ground Truth')
   
   plt.subplot(133)
   plt.imshow(image)
   plt.imshow(pred_mask, alpha=0.5, cmap='Blues')
   plt.title('Prediction')
   
   plt.savefig(f'results/sample_{idx}.png')
   ```

## 常见问题

### 1. 内存不足
问题：CUDA out of memory
解决方案：
```bash
# 1. 减小批次大小
python train.py --batch_size 1

# 2. 使用较小的模型
python train.py --model_type vit_b

# 3. 减小图像分辨率
# 修改 dataset.py 中的 resize 参数
A.Resize(512, 512)
```

### 2. 训练不稳定
问题：损失波动大，难以收敛
解决方案：
```bash
# 1. 降低学习率
python train.py --lr 1e-6

# 2. 增加权重衰减
python train.py --weight_decay 1e-3

# 3. 使用学习率预热
# 修改 train.py 中的学习率调度器
warmup_epochs = 5
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=args.lr,
    epochs=args.epochs,
    steps_per_epoch=len(train_loader),
    pct_start=warmup_epochs/args.epochs
)
```

### 3. 分割效果不佳
问题：分割结果不够精确
解决方案：
```python
# 1. 增加训练轮数
python train.py --epochs 50

# 2. 调整损失函数权重
loss = 0.7 * dice_loss + 0.3 * bce_loss

# 3. 使用多点提示
points = generate_multiple_points(mask)  # 生成多个提示点
```

## 注意事项

1. 硬件要求
   - GPU: NVIDIA GPU with ≥12GB VRAM
   - RAM: ≥16GB
   - Storage: ≥50GB free space

2. 训练建议
   - 定期保存检查点
   - 监控显存使用
   - 记录实验配置
   - 使用验证集调参

3. 代码管理
   - 使用版本控制
   - 记录实验日志
   - 备份重要数据

## 扩展建议

1. 提示策略改进
   ```python
   # 动态提示点生成
   def generate_dynamic_points(mask):
       # 基于掩码形状生成提示点
       return points, labels
   ```

2. 损失函数扩展
   ```python
   # 添加边界损失
   def boundary_loss(pred, target):
       # 计算边界损失
       return loss
   
   # 组合多个损失
   loss = 0.4 * dice_loss + 0.3 * bce_loss + 0.3 * boundary_loss
   ```

3. 集成学习
   ```python
   # 模型集成预测
   def ensemble_predict(models, image):
       predictions = []
       for model in models:
           pred = model(image)
           predictions.append(pred)
       return torch.mean(torch.stack(predictions), dim=0)
   ```

## 参考资料

1. [SAM 官方仓库](https://github.com/facebookresearch/segment-anything)
2. [BUSI 数据集](https://scholar.cu.edu.eg/?q=afahmy/pages/dataset)
3. [SAM 论文](https://arxiv.org/abs/2304.02643)
4. [医学图像分割综述](https://arxiv.org/abs/2103.00591)
5. [Transformer在医学图像分割中的应用](https://arxiv.org/abs/2103.04681)

## 引用

如果您使用了本项目的代码，请引用以下论文：
```bibtex
@article{kirillov2023segany,
  title={Segment Anything},
  author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C. and Lo, Wan-Yen and Doll{\'a}r, Piotr and Girshick, Ross},
  journal={arXiv:2304.02643},
  year={2023}
}
```

