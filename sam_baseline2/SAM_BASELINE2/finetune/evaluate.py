import os
import sys
import argparse
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from segment_anything.build_sam import sam_model_registry
from dataset import build_data_loader
from utils import evaluate_model

def get_args_parser():
    parser = argparse.ArgumentParser('SAM 评估', add_help=False)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--data_path', default='./Dataset_BUSI_with_GT', type=str)
    parser.add_argument('--checkpoint', default='./output/best_model.pth', type=str)
    parser.add_argument('--model_type', default='vit_h', type=str)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--output_dir', default='./evaluation', type=str)
    parser.add_argument('--visualize', action='store_true', help='是否可视化结果')
    parser.add_argument('--num_workers', default=4, type=int)
    
    return parser

def visualize_results(model, data_loader, device, output_dir, num_samples=10):
    """可视化分割结果"""
    os.makedirs(output_dir, exist_ok=True)
    
    model.eval()
    count = 0
    
    with torch.no_grad():
        for batch in data_loader:
            if count >= num_samples:
                break
                
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            image_paths = batch['image_path']
            
            # 获取图像嵌入
            image_embeddings = model.image_encoder(images)
            
            # 生成提示 (使用图像中心点)
            batch_size, _, h, w = images.shape
            points = torch.tensor([[[w//2, h//2]]] * batch_size, device=device).float()
            point_labels = torch.ones(batch_size, 1, device=device)
            
            # 生成掩膜预测
            sparse_embeddings, dense_embeddings = model.prompt_encoder(
                points=points,
                boxes=None,
                masks=None,
            )
            
            mask_predictions, _ = model.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
            
            # 将预测掩码上采样到原始图像尺寸
            mask_predictions = model.postprocess_masks(
                mask_predictions,
                input_size=images.shape[-2:],
                original_size=images.shape[-2:]
            )
            
            # 阈值处理预测掩膜
            pred_masks = (torch.sigmoid(mask_predictions) > 0.5).float()
            
            # 可视化每个样本
            for i in range(images.size(0)):
                if count >= num_samples:
                    break
                    
                img = images[i].cpu().permute(1, 2, 0).numpy()
                img = (img * 255).astype(np.uint8)
                
                gt_mask = masks[i].cpu().numpy()
                pred_mask = pred_masks[i, 0].cpu().numpy()
                
                # 绘制结果
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                axes[0].imshow(img)
                axes[0].set_title('Original Image')
                axes[0].axis('off')
                
                axes[1].imshow(img)
                axes[1].imshow(gt_mask, alpha=0.5, cmap='Reds')
                axes[1].set_title('Ground Truth Mask')
                axes[1].axis('off')
                
                axes[2].imshow(img)
                axes[2].imshow(pred_mask, alpha=0.5, cmap='Blues')
                axes[2].set_title('Predicted Mask')
                axes[2].axis('off')
                
                # 保存结果
                image_name = os.path.basename(image_paths[i])
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'result_{count}_{image_name}'))
                plt.close()
                
                count += 1

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
    sam.to(device)
    
    # 加载检查点
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
        print("生成可视化结果...")
        visualize_results(sam, test_loader, device, 
                         os.path.join(args.output_dir, 'visualizations'))

if __name__ == '__main__':
    args = get_args_parser().parse_args()
    main(args) 