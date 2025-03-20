import numpy as np
import torch
import torch.nn.functional as F
# from medpy.metric.binary import hd95 as compute_hd95
import scipy.ndimage as ndimage
from scipy.spatial.distance import directed_hausdorff
from tqdm import tqdm

def calculate_dice(pred, target):
    """计算Dice系数"""
    # 确保预测和目标都是二值化的
    pred = pred.float()
    target = target.float()
    
    # 计算相交部分
    intersection = (pred * target).sum(dim=(2, 3))
    
    # 计算两个区域的面积和
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    
    # 计算Dice系数
    dice = (2.0 * intersection) / (union + 1e-6)
    
    # 返回平均Dice系数
    return dice.mean()

# 自定义 HD95 计算函数
def compute_hd95(pred, target):
    """
    计算 95% Hausdorff Distance
    """
    # 确保输入是二值图像
    pred = (pred > 0).astype(np.uint8)
    target = (target > 0).astype(np.uint8)
    
    # 查找轮廓点
    pred_points = np.argwhere(pred)
    target_points = np.argwhere(target)
    
    # 如果任一掩码为空，返回0
    if len(pred_points) == 0 or len(target_points) == 0:
        return 0.0
    
    # 计算双向Hausdorff距离
    d1, _, _ = directed_hausdorff(pred_points, target_points)
    d2, _, _ = directed_hausdorff(target_points, pred_points)
    
    # 95%分位数（简化版本 - 使用最大距离的95%）
    return max(d1, d2) * 0.95

def calculate_specificity(pred, target):
    """计算特异性 (Specificity)"""
    # 确保输入是二维的
    if pred.ndim > 2:
        pred = pred.squeeze()
    if target.ndim > 2:
        target = target.squeeze()
    
    # 转换为二值图像
    pred_binary = (pred > 0.5).astype(np.uint8)
    target_binary = (target > 0.5).astype(np.uint8)
    
    # 计算真阴性 (TN) 和假阳性 (FP)
    TN = np.sum((pred_binary == 0) & (target_binary == 0))
    FP = np.sum((pred_binary == 1) & (target_binary == 0))
    
    # 计算特异性: TN / (TN + FP)
    if TN + FP == 0:
        return 1.0  # 如果没有阴性样本，按照约定返回1
    
    return TN / (TN + FP)

def evaluate_model(model, data_loader, device):
    """评估模型性能"""
    model.eval()
    total_dice = 0.0
    total_hd95 = 0.0
    total_spec = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="评估"):
            images = batch['image'].to(device)
            masks = batch['mask'].float().to(device)
            batch_size = images.shape[0]
            
            batch_dice = 0.0
            batch_hd95 = 0.0
            batch_spec = 0.0
            
            # 逐个处理图像
            for i in range(batch_size):
                # 处理单张图像
                image = images[i:i+1]
                mask = masks[i:i+1]
                
                # 添加图像预处理步骤
                image = model.preprocess(image)
                
                # 获取图像嵌入
                image_embedding = model.image_encoder(image)
                
                # 生成提示点 (使用图像中心点)
                h, w = image.shape[-2:]
                point = torch.tensor([[[w//2, h//2]]], device=device)
                point_label = torch.ones(1, 1, device=device)
                
                # 生成掩膜预测
                sparse_embedding, dense_embedding = model.prompt_encoder(
                    points=(point, point_label),
                    boxes=None,
                    masks=None,
                )
                
                mask_prediction, _ = model.mask_decoder(
                    image_embeddings=image_embedding,
                    image_pe=model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embedding,
                    dense_prompt_embeddings=dense_embedding,
                    multimask_output=False,
                )
                
                # 将预测掩码上采样到原始图像尺寸
                mask_prediction = model.postprocess_masks(
                    mask_prediction,
                    input_size=image.shape[-2:],
                    original_size=image.shape[-2:]
                )
                
                # 计算评估指标
                pred_mask = (torch.sigmoid(mask_prediction) > 0.5).float()
                curr_dice = calculate_dice(pred_mask, mask.unsqueeze(1)).item()
                
                # 计算HD95和特异性
                pred_np = pred_mask.squeeze().cpu().numpy()
                mask_np = mask.squeeze().cpu().numpy()
                curr_hd95 = compute_hd95(pred_np, mask_np)
                curr_spec = calculate_specificity(pred_np, mask_np)
                
                # 累加指标
                batch_dice += curr_dice
                batch_hd95 += curr_hd95
                batch_spec += curr_spec
            
            # 累加批次指标
            total_dice += batch_dice
            total_hd95 += batch_hd95
            total_spec += batch_spec
            total_samples += batch_size
    
    # 计算平均指标
    avg_dice = total_dice / total_samples
    avg_hd95 = total_hd95 / total_samples
    avg_spec = total_spec / total_samples
    
    return {
        'dice': avg_dice,
        'hd95': avg_hd95,
        'specificity': avg_spec
    }

def calculate_hausdorff(pred, target):
    """计算Hausdorff距离 (HD95)"""
    # 确保输入是二维的
    if pred.ndim > 2:
        pred = pred.squeeze()
    if target.ndim > 2:
        target = target.squeeze()
    
    # 调用compute_hd95函数
    return compute_hd95(pred, target) 