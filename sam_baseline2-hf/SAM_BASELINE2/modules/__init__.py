"""
高频特征增强模块
用于改进SAM模型在医学图像分割任务中的性能
"""

from .hf_module import HFA_block, add_hf_module

__all__ = ['HFA_block', 'add_hf_module']
