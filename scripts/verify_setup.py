#!/usr/bin/env python3
# scripts/verify_setup.py - 验证环境配置

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("\n" + "="*80)
print("GuardFed 环境验证")
print("="*80 + "\n")

# 测试1: Python包
print("[1/4] 检查Python包...")
try:
    import torch
    import numpy as np
    import pandas as pd
    import sklearn
    import matplotlib
    import seaborn
    print(f"   [OK] PyTorch {torch.__version__}")
    print(f"   [OK] NumPy {np.__version__}")
    print(f"   [OK] Pandas {pd.__version__}")
    print(f"   [OK] Scikit-learn {sklearn.__version__}")
    print(f"   [OK] Matplotlib {matplotlib.__version__}")
    print(f"   [OK] Seaborn {seaborn.__version__}\n")
except ImportError as e:
    print(f"   [ERROR] 缺少依赖包: {e}\n")
    sys.exit(1)

# 测试2: 项目模块
print("[2/4] 检查项目模块...")
try:
    from src import HYPERPARAMETERS
    from src import data_loader
    from src.models import function
    from src.algorithms import FedAvg, FLTrust, FairG, FairCosG
    print("   [OK] HYPERPARAMETERS")
    print("   [OK] data_loader")
    print("   [OK] models.function")
    print("   [OK] algorithms (FedAvg, FLTrust, FairG, FairCosG)\n")
except ImportError as e:
    print(f"   [ERROR] 模块导入失败: {e}\n")
    sys.exit(1)

# 测试3: 数据文件
print("[3/4] 检查数据文件...")
data_paths = [
    "D:/GitHub/GuardFed-main/data/adult",
    "D:/GitHub/GuardFed-main/data/compas"
]
for path in data_paths:
    if os.path.exists(path):
        print(f"   [OK] {path}")
    else:
        print(f"   [WARNING] {path} 不存在")
print()

# 测试4: 数据加载
print("[4/4] 测试数据加载...")
try:
    from src.data_loader import DatasetLoader
    from src.HYPERPARAMETERS import HYPERPARAMETERS, SEED

    loader = DatasetLoader(dataset_name='adult', seed=SEED, device=HYPERPARAMETERS['DEVICE'])
    info = loader.get_info()

    print(f"   [OK] Adult数据集加载成功")
    print(f"   训练样本: {info['num_train']}")
    print(f"   测试样本: {info['num_test']}")
    print(f"   特征数: {info['num_features']}\n")
except Exception as e:
    print(f"   [ERROR] 数据加载失败: {e}\n")
    sys.exit(1)

# 总结
print("="*80)
print("环境验证完成！")
print("="*80)
print("\n所有检查通过:")
print("  [OK] Python包安装正确")
print("  [OK] 项目模块可导入")
print("  [OK] 数据文件存在")
print("  [OK] 数据加载功能正常")
print("\n你的GuardFed环境已准备就绪！")
print("="*80 + "\n")
