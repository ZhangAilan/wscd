#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
计算 TransWCD-SA 模型的参数量
"""

import argparse
import torch
from omegaconf import OmegaConf
from models.model_transwcd_sa import TransWCD_dual, TransWCD_single
from modules.SA_module import build_sg_constraint_from_cfg


def count_parameters(model):
    """计算模型总参数量和可训练参数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def print_param_stats(model, model_name="Model"):
    """打印模型参数统计信息"""
    total_params, trainable_params = count_parameters(model)
    
    print(f"\n{'='*60}")
    print(f"{model_name} 参数统计")
    print(f"{'='*60}")
    print(f"总参数量：     {total_params:,} ({total_params/1e6:.2f} M)")
    print(f"可训练参数量： {trainable_params:,} ({trainable_params/1e6:.2f} M)")
    print(f"{'='*60}\n")
    
    return total_params, trainable_params


def print_layer_stats(model):
    """打印各层参数统计"""
    print("\n各模块参数量详情:")
    print(f"{'-'*60}")
    print(f"{'模块':<40} {'参数量':>15} {'(M)':>10}")
    print(f"{'-'*60}")
    
    total = 0
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        total += params
        print(f"{name:<40} {params:>15,} {params/1e6:>10.2f}")
    
    print(f"{'-'*60}")
    print(f"{'总计':<40} {total:>15,} {total/1e6:>10.2f}")
    print()


def main(args):
    # 加载配置
    cfg = OmegaConf.load(args.config)
    
    # 创建模型
    if cfg.scheme == "transwcd_dual":
        model = TransWCD_dual(
            backbone=cfg.backbone.config,
            stride=cfg.backbone.stride,
            num_classes=cfg.dataset.num_classes,
            embedding_dim=256,
            pretrained=False,  # 不加载预训练权重，只计算结构参数
            pooling=args.pooling
        )
    elif cfg.scheme == "transwcd_single":
        model = TransWCD_single(
            backbone=cfg.backbone.config,
            stride=cfg.backbone.stride,
            num_classes=cfg.dataset.num_classes,
            embedding_dim=256,
            pretrained=False,
            pooling=args.pooling
        )
    else:
        raise ValueError(f"Unknown scheme: {cfg.scheme}")
    
    # 创建 SA 模块
    sa_module = build_sg_constraint_from_cfg(cfg, in_ch=model.in_channels[3])
    
    # 打印主干网络参数
    print(f"\n配置：{args.config}")
    print(f"Scheme: {cfg.scheme}")
    print(f"Backbone: {cfg.backbone.config}")
    
    print_param_stats(model, "TransWCD 主干")
    print_layer_stats(model)
    
    # 打印 SA 模块参数
    print_param_stats(sa_module, "SA Module")
    print_layer_stats(sa_module)
    
    # 总参数量
    total_params = sum(p.numel() for p in model.parameters()) + \
                   sum(p.numel() for p in sa_module.parameters())
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) + \
                      sum(p.numel() for p in sa_module.parameters() if p.requires_grad)
    
    print(f"\n{'='*60}")
    print(f"完整模型 (TransWCD + SA Module) 参数统计")
    print(f"{'='*60}")
    print(f"总参数量：     {total_params:,} ({total_params/1e6:.2f} M)")
    print(f"可训练参数量： {total_trainable:,} ({total_trainable/1e6:.2f} M)")
    print(f"{'='*60}\n")
    
    # 如果提供了 checkpoint 路径，加载并验证
    if args.checkpoint:
        print(f"加载 checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        
        # 创建完整模型用于加载权重
        if cfg.scheme == "transwcd_dual":
            full_model = TransWCD_dual(
                backbone=cfg.backbone.config,
                stride=cfg.backbone.stride,
                num_classes=cfg.dataset.num_classes,
                embedding_dim=256,
                pretrained=False,
                pooling=args.pooling
            )
        else:
            full_model = TransWCD_single(
                backbone=cfg.backbone.config,
                stride=cfg.backbone.stride,
                num_classes=cfg.dataset.num_classes,
                embedding_dim=256,
                pretrained=False,
                pooling=args.pooling
            )
        
        # 加载权重
        missing_keys, unexpected_keys = full_model.load_state_dict(checkpoint, strict=False)
        if missing_keys:
            print(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys}")
        
        print(f"✓ 成功加载 checkpoint")
        
        # 验证参数量
        loaded_params = sum(p.numel() for p in full_model.parameters())
        print(f"加载后模型参数量：{loaded_params:,} ({loaded_params/1e6:.2f} M)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/WHU.yaml", type=str, help="配置文件")
    parser.add_argument("--pooling", default="gmp", type=str, help="pooling method")
    parser.add_argument("--checkpoint", default=None, type=str, help="checkpoint 路径")
    
    args = parser.parse_args()
    main(args)
