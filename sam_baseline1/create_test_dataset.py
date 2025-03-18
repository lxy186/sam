import os
import random
import shutil
from glob import glob
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Create a small test dataset from BUSI')
    parser.add_argument('--src_root', type=str, default='Dataset_BUSI_with_GT', 
                      help='原始数据集路径')
    parser.add_argument('--dst_root', type=str, default='Dataset_BUSI_mini', 
                      help='测试数据集保存路径')
    parser.add_argument('--num_samples', type=int, default=5, 
                      help='每类抽取的样本数量')
    parser.add_argument('--seed', type=int, default=42, 
                      help='随机种子')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 设置随机种子以确保可重复性
    random.seed(args.seed)
    
    # 创建目标目录
    categories = ['benign', 'malignant']
    for category in categories:
        os.makedirs(os.path.join(args.dst_root, category), exist_ok=True)
    
    # 抽取样本
    for category in categories:
        # 获取所有原始图像路径（排除mask文件）
        image_paths = glob(os.path.join(args.src_root, category, f"{category} (*).png"))
        image_paths = [p for p in image_paths if "_mask" not in p]
        
        # 确保样本数量不超过可用数量
        num_to_sample = min(args.num_samples, len(image_paths))
        print(f"Found {len(image_paths)} {category} images, sampling {num_to_sample}")
        
        # 随机抽样
        sampled_paths = random.sample(image_paths, num_to_sample)
        
        # 复制图像和对应的mask
        for img_path in sampled_paths:
            # 目标文件名
            img_filename = os.path.basename(img_path)
            mask_filename = img_filename.replace('.png', '_mask.png')
            
            # 源文件路径
            src_img = img_path
            src_mask = img_path.replace('.png', '_mask.png')
            
            # 目标文件路径
            dst_img = os.path.join(args.dst_root, category, img_filename)
            dst_mask = os.path.join(args.dst_root, category, mask_filename)
            
            # 复制文件
            shutil.copy2(src_img, dst_img)
            shutil.copy2(src_mask, dst_mask)
            print(f"Copied {img_filename} and its mask")
    
    print(f"\nCreated mini dataset at {args.dst_root}")
    print(f"Use it with: python baseline1_sam_zero_shot.py --data_root {args.dst_root}")

if __name__ == "__main__":
    main() 