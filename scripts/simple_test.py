#!/usr/bin/env python3
# scripts/simple_test.py - 简单测试脚本，验证核心模块

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from datetime import datetime

print("\n" + "="*80)
print("GuardFed 简单测试 - 核心模块验证")
print("="*80)
print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80 + "\n")

# ============================================================================
# 测试1: 导入核心模块
# ============================================================================
print("[1/5] 测试模块导入...")
try:
    from src.HYPERPARAMETERS import HYPERPARAMETERS, SEED
    from src.data_loader import DatasetLoader
    from src.models.function import MLP, compute_fairness_metrics, test_inference_modified
    print("   [OK] 所有核心模块导入成功\n")
except Exception as e:
    print(f"   [ERROR] 模块导入失败: {e}\n")
    sys.exit(1)

# ============================================================================
# 测试2: 数据加载
# ============================================================================
print("[2/5] 测试数据加载...")
try:
    data_loader = DatasetLoader(dataset_name='adult', seed=SEED, device=HYPERPARAMETERS['DEVICE'])
    data_info = data_loader.get_info()
    print(f"   [OK] 数据加载成功")
    print(f"   训练样本: {data_info['num_train']}")
    print(f"   测试样本: {data_info['num_test']}")
    print(f"   特征维度: {data_info['num_features']}\n")
except Exception as e:
    print(f"   [ERROR] 数据加载失败: {e}\n")
    sys.exit(1)

# ============================================================================
# 测试3: 模型创建
# ============================================================================
print("[3/5] 测试模型创建...")
try:
    model = MLP(
        num_features=data_info['num_features'],
        num_classes=2,
        seed=SEED
    ).to(HYPERPARAMETERS['DEVICE'])
    print(f"   [OK] 模型创建成功")
    print(f"   模型参数数量: {sum(p.numel() for p in model.parameters())}\n")
except Exception as e:
    print(f"   [ERROR] 模型创建失败: {e}\n")
    sys.exit(1)

# ============================================================================
# 测试4: 模型推理
# ============================================================================
print("[4/5] 测试模型推理...")
try:
    tensors = data_loader.get_tensors()
    X_test = tensors['X_test']
    y_test = tensors['y_test']
    sex_test = tensors['sex_test']

    # 创建测试数据加载器
    test_loader = data_loader.create_test_loader(batch_size=256)

    # 测试推理
    loss, acc, fairness_metrics, per_category = test_inference_modified(
        global_model=model,
        test_loader=test_loader,
        model_class=MLP
    )

    eod = fairness_metrics['EOD']
    spd = fairness_metrics['SPD']

    print(f"   [OK] 模型推理成功")
    print(f"   初始准确率: {acc:.4f}")
    print(f"   初始EOD: {eod:.4f}")
    print(f"   初始SPD: {spd:.4f}\n")
except Exception as e:
    print(f"   [ERROR] 模型推理失败: {e}\n")
    sys.exit(1)

# ============================================================================
# 测试5: 数据分区
# ============================================================================
print("[5/5] 测试数据分区...")
try:
    # 简单的数据分区测试
    num_clients = 6
    num_train = len(tensors['X_train'])
    indices = np.arange(num_train)
    np.random.shuffle(indices)

    # 均匀分配
    client_indices = np.array_split(indices, num_clients)

    print(f"   [OK] 数据分区成功")
    print(f"   客户端数量: {num_clients}")
    print(f"   每个客户端样本数: {[len(idx) for idx in client_indices[:3]]}...\n")
except Exception as e:
    print(f"   [ERROR] 数据分区失败: {e}\n")
    sys.exit(1)

# ============================================================================
# 总结
# ============================================================================
print("="*80)
print("测试完成！所有核心模块工作正常。")
print("="*80)
print("\n核心功能验证:")
print("  [OK] 模块导入")
print("  [OK] 数据加载")
print("  [OK] 模型创建")
print("  [OK] 模型推理")
print("  [OK] 数据分区")
print("\n代码完整性验证通过！")
print("="*80)
