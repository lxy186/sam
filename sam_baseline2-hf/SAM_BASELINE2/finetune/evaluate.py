import os
import sys
import argparse
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import warnings
from typing import Dict, Any
from torch.cuda.amp import autocast

# 添加父目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from segment_anything.build_sam import sam_model_registry
from SAM_BASELINE2.finetune.dataset import build_data_loader
from SAM_BASELINE2.finetune.utils import evaluate_model

# 忽略特定警告
warnings.filterwarnings("ignore", message="Error fetching version info")
warnings.filterwarnings("ignore", category=FutureWarning)

def get_args_parser():
    parser = argparse.ArgumentParser('SAM 评估', add_help=False)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--data_path', default='./Dataset_BUSI_with_GT', type=str)
    parser.add_argument('--checkpoint', default='./output/best_model.pth', type=str)
    parser.add_argument('--model_type', type=str, required=True,
                      choices=['vit_h', 'vit_l', 'vit_b'],
                      help='选择模型类型: vit_h, vit_l, 或 vit_b')
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--output_dir', default='./evaluation', type=str)
    parser.add_argument('--visualize', action='store_true', help='是否可视化结果')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--num_samples', default=10, type=int, help='可视化样本数量')
    parser.add_argument('--use_hf', action='store_true', help='是否使用高频特征增强模块')
    parser.add_argument('--hf_layers', nargs='+', type=int, default=[3,7,11],
                      help='添加HF模块的层索引')
    
    return parser

def visualize_results(model, data_loader, device, output_dir, num_samples=10):
    """可视化分割结果"""
    os.makedirs(output_dir, exist_ok=True)
    
    model.eval()
    count = 0
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="可视化"):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            image_paths = batch['image_path']
            batch_size = images.shape[0]
            
            # 逐个处理图像
            for i in range(batch_size):
                if count >= num_samples:
                    break
                    
                # 处理单张图像
                image = images[i:i+1]
                mask = masks[i:i+1]
                
                # 保存原始图像用于可视化
                orig_img = image[0].cpu().permute(1, 2, 0).numpy()
                
                # 预处理图像
                image = model.preprocess(image)
                
                # 获取图像嵌入
                image_embedding = model.image_encoder(image)
                
                # 生成提示点
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
                
                # 阈值处理预测掩膜
                pred_mask = (torch.sigmoid(mask_prediction) > 0.5).float()
                
                # 准备可视化
                gt_mask = mask[0].cpu().numpy()
                pred_mask = pred_mask[0, 0].cpu().numpy()
                
                # 绘制结果
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                axes[0].imshow(orig_img)
                axes[0].set_title('原始图像')
                axes[0].axis('off')
                
                axes[1].imshow(orig_img)
                axes[1].imshow(gt_mask, alpha=0.5, cmap='Reds')
                axes[1].set_title('真实掩码')
                axes[1].axis('off')
                
                axes[2].imshow(orig_img)
                axes[2].imshow(pred_mask, alpha=0.5, cmap='Blues')
                axes[2].set_title('预测掩码')
                axes[2].axis('off')
                
                # 保存结果
                image_name = os.path.basename(image_paths[i])
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'result_{count}_{image_name}'))
                plt.close()
                
                count += 1
                
            if count >= num_samples:
                break

def main(args):
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 构建数据加载器
    print("构建数据加载器...")
    _, _, test_loader = build_data_loader(
        args.data_path, args.batch_size, args.num_workers
    )
    print(f"测试集大小: {len(test_loader.dataset)}")
    
    # 加载模型
    print(f"加载 SAM 模型 {args.model_type}...")
    sam = sam_model_registry[args.model_type]()
    
    if args.use_hf:
        print("添加高频特征增强模块...")
        from modules.hf_module import add_hf_module
        sam = add_hf_module(sam, layers=args.hf_layers)
    
    sam.to(device)
    
    # 加载最佳模型
    print(f"加载检查点 {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    sam.load_state_dict(checkpoint['model_state_dict'])
    
    # 评估模型
    print("评估模型...")
    metrics = evaluate_model(sam, test_loader, device)
    
    print(f"测试结果:")
    print(f"  Dice系数: {metrics['dice']:.4f}")
    print(f"  HD95: {metrics['hd95']:.4f}")
    print(f"  特异性: {metrics['specificity']:.4f}")
    
    # 保存结果
    with open(os.path.join(args.output_dir, 'results.txt'), 'w') as f:
        f.write(f"Dice系数: {metrics['dice']:.4f}\n")
        f.write(f"HD95: {metrics['hd95']:.4f}\n")
        f.write(f"特异性: {metrics['specificity']:.4f}\n")
    
    # 可视化
    if args.visualize:
        print(f"生成{args.num_samples}张可视化结果...")
        visualize_results(sam, test_loader, device, 
                         os.path.join(args.output_dir, 'visualizations'),
                         num_samples=args.num_samples)

if __name__ == '__main__':
    args = get_args_parser().parse_args()
    main(args) 