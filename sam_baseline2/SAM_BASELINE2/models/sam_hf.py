import torch
import torch.nn as nn
from segment_anything.modeling.sam import Sam
from pytorch_wavelets import DWTForward

class HFA_block(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.wavelet = DWTForward(J=1, wave='db3', mode='zero')
        
        # 高频特征处理
        self.hf_conv = nn.Sequential(
            nn.Conv2d(in_channels*3, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        # 通道注意力
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels//4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels//4, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 小波变换
        LL, Hs = self.wavelet(x)
        # Hs[0] 包含 LH, HL, HH
        HF = torch.cat([Hs[0][:, i, :, :, :] for i in range(3)], dim=1)
        
        # 处理高频特征
        hf_feat = self.hf_conv(HF)
        
        # 通道注意力
        att = self.channel_att(hf_feat)
        
        return x + hf_feat * att

class SAM_HF(Sam):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 添加高频特征增强模块
        self.hf_module = HFA_block(self.image_encoder.patch_embed.proj.out_channels)
        
        # 在图像编码器的特定层后添加HF模块
        self._modify_image_encoder()
    
    def _modify_image_encoder(self):
        # 在特定transformer block后添加HF模块
        original_blocks = self.image_encoder.blocks
        new_blocks = nn.ModuleList()
        
        for i, block in enumerate(original_blocks):
            new_blocks.append(block)
            # 在第4、8、12层后添加HF模块
            if i in [3, 7, 11]:
                new_blocks.append(self.hf_module)
        
        self.image_encoder.blocks = new_blocks

    def forward(self, x, *args, **kwargs):
        return super().forward(x, *args, **kwargs) 