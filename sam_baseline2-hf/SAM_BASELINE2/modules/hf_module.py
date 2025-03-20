import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

class HFA_block(nn.Module):
    """高频特征增强模块"""
    
    def __init__(self, in_channels: int, reduction_ratio: int = 8):
        super().__init__()
        
        # 特征处理
        self.process = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.LayerNorm(in_channels),
            nn.GELU()
        )
        
        # 改进的通道注意力
        self.channel_att = nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.GELU(),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Sigmoid()
        )
        
        # 添加空间注意力
        self.spatial_att = nn.Sequential(
            nn.LayerNorm([in_channels]),
            nn.Linear(in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入 x 的形状是 [B, H, W, C]
        B, H, W, C = x.shape
        
        # 1. 特征处理
        x_processed = self.process(x)
        
        # 2. 通道注意力
        x_pool = x.mean(dim=(1, 2))  # [B, C]
        channel_att = self.channel_att(x_pool).unsqueeze(1).unsqueeze(1)  # [B, 1, 1, C]
        
        # 3. 空间注意力
        x_trans = x.permute(0, 3, 1, 2)  # [B, C, H, W]
        spatial_att = self.spatial_att(x.reshape(B, H*W, C))  # [B, H*W, 1]
        spatial_att = spatial_att.reshape(B, H, W, 1)
        
        # 4. 组合注意力和特征
        enhanced = x + x_processed * channel_att * spatial_att
        
        return enhanced

def add_hf_module(model: nn.Module, 
                 layers: Optional[List[int]] = None,
                 reduction_ratio: int = 8) -> nn.Module:
    """
    动态添加HF模块到模型中
    
    Args:
        model: SAM模型实例
        layers: 需要添加HF模块的层索引列表，默认为[3,7,11]
        reduction_ratio: 通道注意力中的降维比例
    
    Returns:
        添加了HF模块的模型
    """
    if layers is None:
        layers = [3, 7, 11]
        
    # 获取图像编码器的通道数
    in_channels = model.image_encoder.blocks[0].attn.qkv.out_features // 3
    
    # 创建HF模块
    hf_module = HFA_block(in_channels, reduction_ratio)
    
    # 将模块移动到与模型相同的设备上
    if next(model.parameters()).is_cuda:
        hf_module = hf_module.cuda()
    
    # 添加模块
    original_blocks = model.image_encoder.blocks
    new_blocks = nn.ModuleList()
    
    for i, block in enumerate(original_blocks):
        new_blocks.append(block)
        if i in layers:
            new_blocks.append(hf_module)
    
    model.image_encoder.blocks = new_blocks
    return model 