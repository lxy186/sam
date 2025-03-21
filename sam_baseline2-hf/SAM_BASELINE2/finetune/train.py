import os
import sys
from pathlib import Path
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import logging
from torch.cuda.amp import GradScaler
from torch.amp import autocast

# 获取当前文件的绝对路径
current_file = Path(__file__).resolve()
# 获取项目根目录（sam_baseline2）
project_root = current_file.parent.parent.parent
# 将项目根目录添加到Python路径
sys.path.insert(0, str(project_root))

# 现在可以导入segment_anything
from segment_anything.build_sam import sam_model_registry
from SAM_BASELINE2.finetune.dataset import build_data_loader
from SAM_BASELINE2.finetune.utils import calculate_dice, compute_hd95 as calculate_hausdorff, calculate_specificity
from SAM_BASELINE2.modules.hf_module import add_hf_module

# 添加模型配置类
class ModelConfig:
    def __init__(self, embed_dim, num_blocks, mlp_ratio=4):
        self.embed_dim = embed_dim  # 嵌入维度
        self.num_blocks = num_blocks  # 块数
        self.mlp_dim = embed_dim * mlp_ratio  # MLP维度
        self.qkv_dim = embed_dim * 3  # QKV维度
        self.rel_pos_dim = 64  # 相对位置编码维度

# 定义不同模型类型的配置
MODEL_CONFIGS = {
    'vit_b': ModelConfig(embed_dim=768, num_blocks=12),  # ViT-B配置
    'vit_l': ModelConfig(embed_dim=1024, num_blocks=24),  # ViT-L配置
    'vit_h': ModelConfig(embed_dim=1280, num_blocks=32),  # ViT-H配置
}

def get_args_parser():
    parser = argparse.ArgumentParser('SAM 微调', add_help=False)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--lr', type=float, default=1e-5, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--data_path', default='./Dataset_BUSI_with_GT', type=str)
    parser.add_argument('--checkpoint', type=str, required=True,
                      help='模型检查点路径。请确保与选择的model_type匹配：\n'
                           'vit_b: sam_vit_b_01ec64.pth\n'
                           'vit_l: sam_vit_l_0b3195.pth\n'
                           'vit_h: sam_vit_h_4b8939.pth')
    parser.add_argument('--output_dir', default='./output', type=str)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--model_type', type=str, required=True,
                      choices=['vit_h', 'vit_l', 'vit_b', 'sam_hf'],
                      help='选择模型类型: vit_h, vit_l, vit_b, 或 sam_hf')
    parser.add_argument('--validate', action='store_true', help='是否在训练后进行验证')
    parser.add_argument('--use_hf', action='store_true', help='是否使用高频特征增强模块')
    parser.add_argument('--hf_layers', nargs='+', type=int, default=[3,7,11], 
                      help='添加HF模块的层索引')
    
    return parser

def main(args):
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置日志
    logging.basicConfig(
        filename=os.path.join(args.output_dir, 'training.log'),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger()
    console_handler = logging.StreamHandler()
    logger.addHandler(console_handler)
    
    # 构建数据加载器
    logger.info("构建数据加载器...")
    train_loader, val_loader, test_loader = build_data_loader(
        args.data_path, args.batch_size, args.num_workers
    )
    logger.info(f"训练集大小: {len(train_loader.dataset)}")
    logger.info(f"验证集大小: {len(val_loader.dataset)}")
    logger.info(f"测试集大小: {len(test_loader.dataset)}")
    
    # 获取对应模型类型的配置
    model_config = MODEL_CONFIGS.get(args.model_type)
    if not model_config:
        raise ValueError(f"不支持的模型类型: {args.model_type}")
    
    # 处理检查点路径
    if not os.path.isfile(args.checkpoint):
        # 尝试在项目根目录下的 checkpoint 目录查找
        checkpoint_dir = os.path.join(project_root, 'checkpoint')
        checkpoint_path = os.path.join(checkpoint_dir, os.path.basename(args.checkpoint))
        if os.path.isfile(checkpoint_path):
            args.checkpoint = checkpoint_path
            logger.info(f"找到检查点文件: {checkpoint_path}")
        else:
            raise FileNotFoundError(f"找不到检查点文件: {args.checkpoint}\n"
                                  f"也不在默认路径: {checkpoint_path}")
    
    # 加载模型
    logger.info(f"加载 SAM 模型 {args.model_type}...")
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    
    # 验证模型结构是否符合预期
    image_encoder = sam.image_encoder
    actual_embed_dim = image_encoder.patch_embed.proj.out_channels
    actual_num_blocks = len(image_encoder.blocks)
    
    if actual_embed_dim != model_config.embed_dim or actual_num_blocks != model_config.num_blocks:
        logger.error(f"模型结构不匹配!")
        logger.error(f"期望配置: embed_dim={model_config.embed_dim}, num_blocks={model_config.num_blocks}")
        logger.error(f"实际配置: embed_dim={actual_embed_dim}, num_blocks={actual_num_blocks}")
        raise ValueError("模型结构与所选类型不匹配")
    
    # 添加HF模块
    if args.use_hf:
        logger.info("添加高频特征增强模块...")
        sam = add_hf_module(sam, layers=args.hf_layers)
        logger.info(f"在层 {args.hf_layers} 添加了HF模块")
    
    sam.to(device)
    
    # 冻结图像编码器
    logger.info("冻结图像编码器参数...")
    for param in sam.image_encoder.parameters():
        param.requires_grad = False
    
    # 统计可训练参数
    total_params = sum(p.numel() for p in sam.parameters())
    trainable_params = sum(p.numel() for p in sam.parameters() if p.requires_grad)
    logger.info(f"总参数量: {total_params/1e6:.2f}M")
    logger.info(f"可训练参数量: {trainable_params/1e6:.2f}M ({100*trainable_params/total_params:.2f}%)")
    
    # 定义优化器
    optimizer = optim.AdamW(
        [p for p in sam.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # 学习率调度器
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    
    # TensorBoard 日志
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'tensorboard'))
    
    # 定义损失函数
    dice_loss = lambda pred, target: 1 - calculate_dice(pred, target)
    bce_loss = nn.BCEWithLogitsLoss()
    
    # 添加进度输出函数
    def print_metrics(epoch, epochs, phase, loss, dice, lr=None, hd95=None, spec=None):
        """打印训练/验证指标"""
        progress = f"[Epoch {epoch}/{epochs}] {phase:^10}"
        metrics = f"Loss: {loss:.4f} | Dice: {dice:.4f}"
        
        if hd95 is not None:
            metrics += f" | HD95: {hd95:.4f}"
        if spec is not None:
            metrics += f" | Spec: {spec:.4f}"
        if lr is not None:
            metrics += f" | LR: {lr:.6f}"
        
        print(f"{progress} | {metrics}")
    
    # 训练循环
    best_val_dice = 0.0
    for epoch in range(args.epochs):
        logger.info(f"开始 Epoch {epoch+1}/{args.epochs}")
        
        # 训练
        sam.train()
        train_loss = 0.0
        train_dice = 0.0
        
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} Training")
        for batch in train_bar:
            images = batch['image'].to(device)
            masks = batch['mask'].float().to(device)
            batch_size = images.shape[0]
            
            # 所有批次的预测和损失
            all_loss = 0
            all_dice = 0
            
            # 逐图像处理
            for i in range(batch_size):
                with autocast('cuda', enabled=True):
                    # 处理单张图像
                    image = images[i:i+1]  # 保持批次维度
                    mask = masks[i:i+1]    # 保持批次维度
                    
                    # 使用SAM的预处理方法
                    image = sam.preprocess(image)
                    
                    # 获取图像特征
                    image_embedding = sam.image_encoder(image)
                    
                    # 生成提示点 (使用图像中心点)
                    h, w = image.shape[-2:]
                    point = torch.tensor([[[w//2, h//2]]], device=device)
                    point_label = torch.ones(1, 1, device=device)
                    
                    # 生成掩膜预测
                    sparse_embedding, dense_embedding = sam.prompt_encoder(
                        points=(point, point_label),
                        boxes=None,
                        masks=None,
                    )
                    
                    # 预测掩膜
                    mask_prediction, _ = sam.mask_decoder(
                        image_embeddings=image_embedding,
                        image_pe=sam.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embedding,
                        dense_prompt_embeddings=dense_embedding,
                        multimask_output=False,
                    )
                    
                    # 将预测掩码上采样到原始图像尺寸
                    mask_prediction = sam.postprocess_masks(
                        mask_prediction,
                        input_size=image.shape[-2:],  # 输入图像尺寸
                        original_size=image.shape[-2:] # 原始图像尺寸
                    )
                    
                    # 计算损失
                    loss_dice = dice_loss(torch.sigmoid(mask_prediction), mask.unsqueeze(1))
                    loss_bce = bce_loss(mask_prediction, mask.unsqueeze(1))
                    loss = 0.5 * loss_dice + 0.5 * loss_bce
                    
                    # 计算当前图像的Dice
                    pred_mask = (torch.sigmoid(mask_prediction) > 0.5).float()
                    curr_dice = calculate_dice(pred_mask, mask.unsqueeze(1)).item()
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # 累加损失和指标
                all_loss += loss.item()
                all_dice += curr_dice
            
            # 记录批次平均损失和指标
            avg_loss = all_loss / batch_size
            avg_dice = all_dice / batch_size
            train_loss += avg_loss
            train_dice += avg_dice
            
            # 更新进度条
            train_bar.set_postfix({
                'loss': f"{avg_loss:.4f}", 
                'dice': f"{avg_dice:.4f}",
                'lr': f"{optimizer.param_groups[0]['lr']:.6f}"
            })
        
        # 更新学习率
        lr_scheduler.step()
        
        # 计算平均指标
        train_loss /= len(train_loader)
        train_dice /= len(train_loader)
        
        # 记录训练指标
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Dice/train', train_dice, epoch)
        
        # 在终端打印摘要
        print_metrics(epoch+1, args.epochs, "Training", train_loss, train_dice, 
                     lr=optimizer.param_groups[0]['lr'])
    
    if args.validate:
        # 训练完成后进行验证
        logger.info("开始验证...")
        sam.eval()
        val_loss = 0.0
        val_dice = 0.0
        val_hd95 = 0.0
        val_spec = 0.0

        with torch.no_grad():
            val_bar = tqdm(val_loader, desc="Validation")
            for batch in val_bar:
                images = batch['image'].to(device)
                masks = batch['mask'].to(device)
                batch_size = images.shape[0]
                
                batch_loss = 0
                batch_dice = 0
                batch_hd95 = 0
                batch_spec = 0
                
                # 逐图像处理验证集
                for i in range(batch_size):
                    # 处理单张图像
                    image = images[i:i+1]
                    mask = masks[i:i+1].float()
                    
                    # 使用SAM的预处理方法
                    image = sam.preprocess(image)
                    
                    # 获取图像特征
                    image_embedding = sam.image_encoder(image)
                    
                    # 生成提示点
                    h, w = image.shape[-2:]
                    point = torch.tensor([[[w//2, h//2]]], device=device)
                    point_label = torch.ones(1, 1, device=device)
                    
                    # 生成掩膜预测
                    sparse_embedding, dense_embedding = sam.prompt_encoder(
                        points=(point, point_label),
                        boxes=None,
                        masks=None,
                    )
                    
                    # 预测掩膜
                    mask_prediction, _ = sam.mask_decoder(
                        image_embeddings=image_embedding,
                        image_pe=sam.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embedding,
                        dense_prompt_embeddings=dense_embedding,
                        multimask_output=False,
                    )
                    
                    # 将预测掩码上采样到原始图像尺寸
                    mask_prediction = sam.postprocess_masks(
                        mask_prediction,
                        input_size=image.shape[-2:],
                        original_size=image.shape[-2:]
                    )
                    
                    # 计算损失
                    loss_dice = dice_loss(torch.sigmoid(mask_prediction), mask.unsqueeze(1))
                    loss_bce = bce_loss(mask_prediction, mask.unsqueeze(1))
                    loss = 0.5 * loss_dice + 0.5 * loss_bce
                    
                    # 计算评估指标
                    pred_mask = (torch.sigmoid(mask_prediction) > 0.5).float()
                    curr_dice = calculate_dice(pred_mask, mask.unsqueeze(1)).item()
                    
                    # 计算HD95和特异性
                    pred_np = pred_mask.squeeze().cpu().numpy()
                    mask_np = mask.squeeze().cpu().numpy()
                    curr_hd95 = calculate_hausdorff(pred_np, mask_np)
                    curr_spec = calculate_specificity(pred_np, mask_np)
                    
                    # 累加指标
                    batch_loss += loss.item()
                    batch_dice += curr_dice
                    batch_hd95 += curr_hd95
                    batch_spec += curr_spec
                
                # 记录批次平均指标
                val_loss += batch_loss / batch_size
                val_dice += batch_dice / batch_size
                val_hd95 += batch_hd95
                val_spec += batch_spec
                
                # 更新进度条
                val_bar.set_postfix({
                    'loss': f"{batch_loss/batch_size:.4f}", 
                    'dice': f"{batch_dice/batch_size:.4f}"
                })

        # 计算平均指标
        val_loss /= len(val_loader)
        val_dice /= len(val_loader)
        val_hd95 /= len(val_loader.dataset)
        val_spec /= len(val_loader.dataset)

        # 记录验证指标
        writer.add_scalar('Loss/val', val_loss, args.epochs-1)
        writer.add_scalar('Dice/val', val_dice, args.epochs-1)
        writer.add_scalar('HD95/val', val_hd95, args.epochs-1)
        writer.add_scalar('Specificity/val', val_spec, args.epochs-1)

        # 在终端打印验证摘要
        print_metrics(args.epochs, args.epochs, "Validation", val_loss, val_dice, 
                     hd95=val_hd95, spec=val_spec)

        # 保存最佳模型
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save({
                'epoch': args.epochs-1,
                'model_state_dict': sam.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dice': val_dice,
                'val_hd95': val_hd95,
                'val_spec': val_spec,
            }, os.path.join(args.output_dir, 'best_model.pth'))
            print(f"\n保存最佳模型 -> 验证 Dice: {val_dice:.4f}, HD95: {val_hd95:.4f}, Spec: {val_spec:.4f}\n")
        
        # 测试最佳模型
        logger.info("加载最佳模型进行测试...")
        checkpoint = torch.load(os.path.join(args.output_dir, 'best_model.pth'))
        sam.load_state_dict(checkpoint['model_state_dict'])
        
        sam.eval()
        test_dice = 0.0
        test_hd95 = 0.0
        test_spec = 0.0
        test_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing"):
                images = batch['image'].to(device)
                masks = batch['mask'].to(device)
                batch_size = images.shape[0]
                
                batch_dice = 0
                batch_hd95 = 0
                batch_spec = 0
                
                # 逐图像处理测试集
                for i in range(batch_size):
                    # 处理单张图像
                    image = images[i:i+1]
                    mask = masks[i:i+1].float()
                    
                    # 使用SAM的预处理方法
                    image = sam.preprocess(image)
                    
                    # 获取图像特征
                    image_embedding = sam.image_encoder(image)
                    
                    # 生成提示点
                    h, w = image.shape[-2:]
                    point = torch.tensor([[[w//2, h//2]]], device=device)
                    point_label = torch.ones(1, 1, device=device)
                    
                    # 生成掩膜预测
                    sparse_embedding, dense_embedding = sam.prompt_encoder(
                        points=(point, point_label),
                        boxes=None,
                        masks=None,
                    )
                    
                    # 预测掩膜
                    mask_prediction, _ = sam.mask_decoder(
                        image_embeddings=image_embedding,
                        image_pe=sam.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embedding,
                        dense_prompt_embeddings=dense_embedding,
                        multimask_output=False,
                    )
                    
                    # 将预测掩码上采样到原始图像尺寸
                    mask_prediction = sam.postprocess_masks(
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
                    curr_hd95 = calculate_hausdorff(pred_np, mask_np)
                    curr_spec = calculate_specificity(pred_np, mask_np)
                    
                    # 累加指标
                    batch_dice += curr_dice
                    batch_hd95 += curr_hd95
                    batch_spec += curr_spec
                
                # 累加批次指标
                test_dice += batch_dice
                test_hd95 += batch_hd95
                test_spec += batch_spec
                test_samples += batch_size
            
            # 计算平均指标
            test_dice /= test_samples
            test_hd95 /= test_samples
            test_spec /= test_samples
        
        logger.info(f"测试结果 - Dice: {test_dice:.4f}, HD95: {test_hd95:.4f}, Spec: {test_spec:.4f}")
    
    # 关闭 TensorBoard writer
    writer.close()

if __name__ == '__main__':
    args = get_args_parser().parse_args()
    main(args) 