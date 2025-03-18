import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from glob import glob
import json
from datetime import datetime
import random
import argparse
import time
from collections import defaultdict

# 导入SAM模型相关组件
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# 参数设置
def parse_args():
    parser = argparse.ArgumentParser(description='SAM Zero-shot Segmentation for BUSI Dataset')
    parser.add_argument('--data_root', type=str, default='Dataset_BUSI_with_GT', help='数据集根目录')
    parser.add_argument('--checkpoint', type=str, default='checkpoint/sam_vit_h_4b8939.pth', help='SAM模型检查点路径')
    parser.add_argument('--result_dir', type=str, default='results_baseline1', help='结果保存目录')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='训练集比例')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--batch_size', type=int, default=32, help='批处理图像数量')
    parser.add_argument('--points_per_side', type=int, default=32, help='SAM每边采样点数量，恢复为原始默认值')
    parser.add_argument('--pred_iou_thresh', type=float, default=0.8, help='SAM预测IoU阈值')
    parser.add_argument('--stability_score_thresh', type=float, default=0.9, help='SAM稳定性分数阈值')
    parser.add_argument('--crop_n_layers', type=int, default=1, help='裁剪层数')
    parser.add_argument('--crop_n_points_downscale_factor', type=int, default=2, help='裁剪点下采样因子')
    parser.add_argument('--min_mask_region_area', type=int, default=100, help='最小mask区域面积')
    parser.add_argument('--workers', type=int, default=8, help='数据加载线程数')
    parser.add_argument('--save_all_masks', action='store_true', help='是否保存所有生成的mask')
    parser.add_argument('--verbose', action='store_true', help='是否打印详细信息')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载的工作进程数')
    parser.add_argument('--pin_memory', action='store_true', help='使用pin_memory加速数据加载')
    parser.add_argument('--image_size', type=int, default=1024, help='处理图像的大小')
    return parser.parse_args()

# 评估指标函数
def calculate_dice(pred_mask, gt_mask):
    """计算Dice系数"""
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = pred_mask.sum() + gt_mask.sum()
    if union == 0:
        return 1.0  # 如果两个mask都是空的，视为完全重合
    return 2.0 * intersection / union

def calculate_iou(pred_mask, gt_mask):
    """计算IoU (Intersection over Union)"""
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    if union == 0:
        return 1.0  # 如果两个mask都是空的，视为完全重合
    return intersection / union

# 计算Hausdorff距离
def calculate_hausdorff(pred_mask, gt_mask):
    """计算Hausdorff距离"""
    from scipy.ndimage import distance_transform_edt
    
    # 如果掩码为空，返回最大距离
    if not np.any(pred_mask) or not np.any(gt_mask):
        return float('inf')
    
    # 计算二进制边缘
    pred_contour = pred_mask ^ np.logical_and(
        pred_mask, 
        np.logical_and(
            np.logical_and(
                np.roll(pred_mask, 1, axis=0), 
                np.roll(pred_mask, -1, axis=0)
            ),
            np.logical_and(
                np.roll(pred_mask, 1, axis=1), 
                np.roll(pred_mask, -1, axis=1)
            )
        )
    )
    
    gt_contour = gt_mask ^ np.logical_and(
        gt_mask, 
        np.logical_and(
            np.logical_and(
                np.roll(gt_mask, 1, axis=0), 
                np.roll(gt_mask, -1, axis=0)
            ),
            np.logical_and(
                np.roll(gt_mask, 1, axis=1), 
                np.roll(gt_mask, -1, axis=1)
            )
        )
    )
    
    # 计算距离变换
    pred_distance = distance_transform_edt(~pred_contour)
    gt_distance = distance_transform_edt(~gt_contour)
    
    # 计算Hausdorff距离
    hausdorff_gt_to_pred = np.max(pred_distance[gt_contour])
    hausdorff_pred_to_gt = np.max(gt_distance[pred_contour])
    
    # 双向Hausdorff距离
    hausdorff = max(hausdorff_gt_to_pred, hausdorff_pred_to_gt)
    
    return hausdorff

# 数据加载函数
def load_image_and_mask(image_path, image_size):
    """优化图像加载函数"""
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 使用更大的目标尺寸以保持更多细节
    target_size = (image_size, image_size)
    image = cv2.resize(image, target_size)
    
    mask_path = image_path.replace('.png', '_mask.png')
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, target_size)
    mask = mask > 0
    
    return image, mask

def visualize_and_save(image, gt_mask, pred_mask, save_path, category, split):
    """可视化和保存结果"""
    plt.figure(figsize=(15, 5))
    
    # 原始图像
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')
    
    # 真实标签
    plt.subplot(1, 3, 2)
    plt.imshow(image)
    plt.imshow(gt_mask, alpha=0.5, cmap='Reds')
    plt.title('Ground Truth')
    plt.axis('off')
    
    # 预测结果
    plt.subplot(1, 3, 3)
    plt.imshow(image)
    plt.imshow(pred_mask, alpha=0.5, cmap='Blues')
    plt.title('SAM Prediction')
    plt.axis('off')
    
    plt.suptitle(f"{category} - {split}")
    plt.tight_layout()
    # 创建子目录
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def split_dataset(image_paths, train_ratio, seed):
    """将数据集分为训练集和验证集"""
    random.seed(seed)
    random.shuffle(image_paths)
    split_idx = int(len(image_paths) * train_ratio)
    return image_paths[:split_idx], image_paths[split_idx:]

def process_batch(mask_generator, images, gt_masks, paths, category, split, args, results, metrics_tracker):
    """批量处理图像"""
    batch_size = len(images)
    
    try:
        # 将图像转换为numpy数组并确保所有图像具有相同的大小
        images_array = np.stack(images)  # shape: (B, H, W, 3)
        
        # 对于32G显存，我们可以一次处理更多图像
        chunk_size = args.batch_size  # 可以根据显存大小调整
        all_masks = []
        
        for i in range(0, batch_size, chunk_size):
            chunk_images = images_array[i:i+chunk_size]
            chunk_masks = []
            
            # 并行处理一个chunk中的所有图像
            for img in chunk_images:
                masks = mask_generator.generate(img)
                if masks:
                    best_mask_idx = np.argmax([mask['predicted_iou'] for mask in masks])
                    best_mask = masks[best_mask_idx]['segmentation'].astype(bool)
                else:
                    best_mask = np.zeros_like(gt_masks[i])
                chunk_masks.append(best_mask)
            
            all_masks.extend(chunk_masks)
        
        batch_results = []
        start_time = time.time()
        
        # 处理每张图像的结果
        for i in range(batch_size):
            image = images[i]
            gt_mask = gt_masks[i]
            pred_mask = all_masks[i]
            image_path = paths[i]
            img_name = os.path.basename(image_path).replace('.png', '')
            
            # 计算评估指标
            dice = calculate_dice(pred_mask, gt_mask)
            iou = calculate_iou(pred_mask, gt_mask)
            try:
                hausdorff = calculate_hausdorff(pred_mask, gt_mask)
            except:
                hausdorff = float('nan')
            
            # 保存每张图像的详细结果
            img_result = {
                'filename': img_name,
                'category': category,
                'split': split,
                'metrics': {
                    'dice': float(dice),
                    'iou': float(iou),
                    'hausdorff': float(hausdorff) if not np.isnan(hausdorff) else "NaN",
                    'pred_iou': float(iou),
                    'num_masks': int(pred_mask.sum())
                },
                'image_shape': image.shape[:2],
                'mask_size': int(pred_mask.sum())
            }
            batch_results.append(img_result)
            
            # 更新结果
            results[split][category]['dice'].append(dice)
            results[split][category]['iou'].append(iou)
            results[split]['overall']['dice'].append(dice)
            results[split]['overall']['iou'].append(iou)
            
            # 更新指标跟踪器
            metrics_tracker[split][category]['dice'].append(dice)
            metrics_tracker[split][category]['iou'].append(iou)
            metrics_tracker[split]['overall']['dice'].append(dice)
            metrics_tracker[split]['overall']['iou'].append(iou)
            
            # 可视化并保存结果
            save_dir = os.path.join(args.result_dir, split, category)
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{img_name}_results.png")
            
            visualize_and_save(image, gt_mask, pred_mask, save_path, category, split)
            
            if args.verbose:
                print(f"  {img_name}: Dice={dice:.4f}, IoU={iou:.4f}, NumMasks={int(pred_mask.sum())}")
        
        # 计算批次处理时间
        elapsed = time.time() - start_time
        print(f"Processed batch of {batch_size} images in {elapsed:.2f}s ({batch_size/elapsed:.2f} imgs/s)")
        
        # 打印当前进度的平均指标
        for cat in ['overall'] + list(set([r['category'] for r in batch_results])):
            if cat == 'overall':
                dice_values = metrics_tracker[split]['overall']['dice']
                iou_values = metrics_tracker[split]['overall']['iou']
            else:
                dice_values = metrics_tracker[split][cat]['dice']
                iou_values = metrics_tracker[split][cat]['iou']
            
            print(f"  Current {split}/{cat}: Dice={np.mean(dice_values):.4f}, IoU={np.mean(iou_values):.4f} ({len(dice_values)} images)")
        
        return results, batch_results
    
    except Exception as e:
        print(f"Error processing batch: {str(e)}")
        return results, []

def main():
    # 解析参数
    args = parse_args()
    
    # 设置随机种子以确保可重复性
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        # 启用CUDA优化
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    # 创建结果目录
    os.makedirs(args.result_dir, exist_ok=True)
    
    # 保存实验配置
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_file = os.path.join(args.result_dir, f"config_{timestamp}.json")
    with open(config_file, 'w') as f:
        json.dump(vars(args), f, indent=2)
    print(f"实验配置已保存到 {config_file}")
    
    # 1. 加载SAM模型
    print("============ SAM Zero-shot Segmentation ============")
    print(f"Loading SAM model from {args.checkpoint}...")
    start_time = time.time()
    sam = sam_model_registry["vit_h"](checkpoint=args.checkpoint)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam.to(device=device)
    print(f"Model loaded successfully in {time.time()-start_time:.2f}s! Using device: {device}")
    
    # 2. 创建自动mask生成器（用于零样本推理）
    print("\n============ SAM Configuration ============")
    print(f"Points per side: {args.points_per_side}")
    print(f"Pred IoU threshold: {args.pred_iou_thresh}")
    print(f"Stability score threshold: {args.stability_score_thresh}")
    print(f"Min mask region area: {args.min_mask_region_area}")
    print(f"Crop n layers: {args.crop_n_layers}")
    print(f"Crop n points downscale factor: {args.crop_n_points_downscale_factor}")
    
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=args.points_per_side,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_score_thresh,
        crop_n_layers=args.crop_n_layers,
        crop_n_points_downscale_factor=args.crop_n_points_downscale_factor,
        min_mask_region_area=args.min_mask_region_area,
        output_mode="binary_mask",
    )
    
    # 3. 准备数据集
    print("\n============ Dataset Information ============")
    categories = ['benign', 'malignant']
    all_images = {
        'benign': [],
        'malignant': []
    }
    
    # 加载数据集信息
    for category in categories:
        image_paths = glob(os.path.join(args.data_root, category, f"{category} (*).png"))
        image_paths = [p for p in image_paths if "_mask" not in p]  # 过滤掉mask文件
        all_images[category] = image_paths
        print(f"Found {len(image_paths)} {category} images")
    
    # 分割数据集
    train_images = {}
    val_images = {}
    for category in categories:
        train_imgs, val_imgs = split_dataset(all_images[category], args.train_ratio, args.seed)
        train_images[category] = train_imgs
        val_images[category] = val_imgs
        print(f"{category}: {len(train_imgs)} training images, {len(val_imgs)} validation images")
    
    # 4. 结果存储结构
    results = {
        'train': {
            'overall': {'dice': [], 'iou': []},
            'benign': {'dice': [], 'iou': []},
            'malignant': {'dice': [], 'iou': []}
        },
        'val': {
            'overall': {'dice': [], 'iou': []},
            'benign': {'dice': [], 'iou': []},
            'malignant': {'dice': [], 'iou': []}
        }
    }
    
    # 指标跟踪器 - 用于实时显示进度
    metrics_tracker = {
        'train': {
            'overall': {'dice': [], 'iou': []},
            'benign': {'dice': [], 'iou': []},
            'malignant': {'dice': [], 'iou': []}
        },
        'val': {
            'overall': {'dice': [], 'iou': []},
            'benign': {'dice': [], 'iou': []},
            'malignant': {'dice': [], 'iou': []}
        }
    }
    
    # 保存每张图像的详细结果
    detailed_results = []
    
    # 5. 处理训练集和验证集
    splits = ['train', 'val']
    total_start_time = time.time()
    
    for split in splits:
        split_start_time = time.time()
        print(f"\n============ Processing {split} set ============")
        image_set = train_images if split == 'train' else val_images
        
        for category in categories:
            cat_start_time = time.time()
            print(f"Processing {category} images...")
            
            # 创建进度条
            total_images = len(image_set[category])
            progress_bar = tqdm(total=total_images, desc=f"{split}/{category}")
            
            # 批量处理图像
            batch_images = []
            batch_gt_masks = []
            batch_paths = []
            
            for idx, image_path in enumerate(image_set[category]):
                try:
                    # 加载图像和真实标签，传入image_size参数
                    image, gt_mask = load_image_and_mask(image_path, args.image_size)
                    
                    # 添加到批次
                    batch_images.append(image)
                    batch_gt_masks.append(gt_mask)
                    batch_paths.append(image_path)
                    
                    # 当批次填满或到达最后一张图像时处理
                    if len(batch_images) == args.batch_size or idx == len(image_set[category]) - 1:
                        results, batch_results = process_batch(
                            mask_generator, batch_images, batch_gt_masks, 
                            batch_paths, category, split, args, results, metrics_tracker
                        )
                        
                        # 添加到详细结果
                        detailed_results.extend(batch_results)
                        
                        # 更新进度条
                        progress_bar.update(len(batch_images))
                        
                        # 清空批次
                        batch_images = []
                        batch_gt_masks = []
                        batch_paths = []
                        
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")
                    # 更新进度条
                    progress_bar.update(1)
            
            progress_bar.close()
            print(f"Finished processing {category} images in {time.time()-cat_start_time:.2f}s")
        
        print(f"Finished processing {split} set in {time.time()-split_start_time:.2f}s")
    
    total_elapsed = time.time() - total_start_time
    print(f"\nTotal processing time: {total_elapsed:.2f}s")
    
    # 6. 汇总并保存结果
    summary = {}
    print("\n============ Results Summary ============")
    for split in splits:
        summary[split] = {}
        print(f"\n{split.upper()} SET RESULTS:")
        
        for category in ['overall'] + categories:
            dice_values = results[split][category]['dice']
            iou_values = results[split][category]['iou']
            
            if not dice_values:  # 如果没有数据
                continue
                
            summary[split][category] = {
                'mean_dice': float(np.mean(dice_values)),
                'std_dice': float(np.std(dice_values)),
                'median_dice': float(np.median(dice_values)),
                'max_dice': float(np.max(dice_values)),
                'min_dice': float(np.min(dice_values)),
                
                'mean_iou': float(np.mean(iou_values)),
                'std_iou': float(np.std(iou_values)),
                'median_iou': float(np.median(iou_values)),
                'max_iou': float(np.max(iou_values)),
                'min_iou': float(np.min(iou_values)),
                
                'num_samples': len(dice_values)
            }
            
            print(f"{category.capitalize()}:")
            print(f"  Samples: {summary[split][category]['num_samples']}")
            print(f"  Mean Dice: {summary[split][category]['mean_dice']:.4f} ± {summary[split][category]['std_dice']:.4f}")
            print(f"  Median Dice: {summary[split][category]['median_dice']:.4f} (Min: {summary[split][category]['min_dice']:.4f}, Max: {summary[split][category]['max_dice']:.4f})")
            print(f"  Mean IoU:  {summary[split][category]['mean_iou']:.4f} ± {summary[split][category]['std_iou']:.4f}")
            print(f"  Median IoU:  {summary[split][category]['median_iou']:.4f} (Min: {summary[split][category]['min_iou']:.4f}, Max: {summary[split][category]['max_iou']:.4f})")
    
    # 生成更完整的结果JSON
    final_results = {
        'metadata': {
            'timestamp': timestamp,
            'runtime': total_elapsed,
            'args': vars(args),
            'device': device,
            'sam_version': 'vit_h',
        },
        'summary': summary,
        'detailed_results': detailed_results
    }
    
    # 保存结果到JSON文件
    results_file = os.path.join(args.result_dir, f"results_summary_{timestamp}.json")
    with open(results_file, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\nResults saved to {results_file}")
    
    # 7. 生成结果可视化
    try:
        # 创建条形图比较benign和malignant的Dice和IoU
        plt.figure(figsize=(12, 10))
        
        # Dice比较
        plt.subplot(2, 1, 1)
        
        x = np.arange(len(categories))
        width = 0.35
        
        train_dice = [summary['train'][cat]['mean_dice'] for cat in categories]
        train_dice_std = [summary['train'][cat]['std_dice'] for cat in categories]
        
        val_dice = [summary['val'][cat]['mean_dice'] for cat in categories]
        val_dice_std = [summary['val'][cat]['std_dice'] for cat in categories]
        
        plt.bar(x - width/2, train_dice, width, label='Train', yerr=train_dice_std, capsize=5)
        plt.bar(x + width/2, val_dice, width, label='Val', yerr=val_dice_std, capsize=5)
        
        plt.ylabel('Dice Coefficient')
        plt.title('Dice Coefficient by Category')
        plt.xticks(x, categories)
        plt.legend()
        
        # IoU比较
        plt.subplot(2, 1, 2)
        
        train_iou = [summary['train'][cat]['mean_iou'] for cat in categories]
        train_iou_std = [summary['train'][cat]['std_iou'] for cat in categories]
        
        val_iou = [summary['val'][cat]['mean_iou'] for cat in categories]
        val_iou_std = [summary['val'][cat]['std_iou'] for cat in categories]
        
        plt.bar(x - width/2, train_iou, width, label='Train', yerr=train_iou_std, capsize=5)
        plt.bar(x + width/2, val_iou, width, label='Val', yerr=val_iou_std, capsize=5)
        
        plt.ylabel('IoU')
        plt.title('IoU by Category')
        plt.xticks(x, categories)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(args.result_dir, f"metrics_comparison_{timestamp}.png"))
        plt.close()
        
        print(f"Visualization saved to {args.result_dir}/metrics_comparison_{timestamp}.png")
    except Exception as e:
        print(f"Error generating visualization: {e}")

if __name__ == "__main__":
    main() 