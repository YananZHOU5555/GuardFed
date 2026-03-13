#!/usr/bin/env python3
# scripts/reproduce_table4_simple.py - 简化版本，完全独立实现

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import copy
from datetime import datetime

print("\n" + "="*80)
print("复现论文表4 - Adult数据集 (简化独立版本)")
print("="*80)
print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80 + "\n")

# ============================================================================
# 配置
# ============================================================================
SEED = 123
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device('cpu')
DATASET_NAME = 'adult'
NUM_ROUNDS = 5
NUM_CLIENTS = 20
NUM_MALICIOUS = 4
BATCH_SIZE = 256
LEARNING_RATE = 0.01

# ============================================================================
# 简单的MLP模型
# ============================================================================
class SimpleMLP(nn.Module):
    def __init__(self, input_size, output_size=2):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# ============================================================================
# 评估函数
# ============================================================================
def evaluate_model(model, test_loader, device):
    """评估模型"""
    model.eval()
    correct = 0
    total = 0

    # 用于计算公平性指标
    predictions = []
    true_labels = []
    sensitive_attrs = []

    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 3:
                X, y, sex = batch
            else:
                X, y = batch
                sex = torch.zeros(len(X), dtype=torch.long)

            X, y = X.to(device), y.to(device)

            outputs = model(X)
            _, predicted = torch.max(outputs.data, 1)

            total += y.size(0)
            correct += (predicted == y).sum().item()

            predictions.extend(predicted.cpu().numpy())
            true_labels.extend(y.cpu().numpy())
            sensitive_attrs.extend(sex.numpy())

    accuracy = correct / total

    # 计算公平性指标
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)
    sensitive_attrs = np.array(sensitive_attrs)

    # AEOD (Equal Opportunity Difference)
    # TPR for group 0 (e.g., female)
    mask_0_pos = (sensitive_attrs == 0) & (true_labels == 1)
    if mask_0_pos.sum() > 0:
        tpr_0 = (predictions[mask_0_pos] == 1).sum() / mask_0_pos.sum()
    else:
        tpr_0 = 0

    # TPR for group 1 (e.g., male)
    mask_1_pos = (sensitive_attrs == 1) & (true_labels == 1)
    if mask_1_pos.sum() > 0:
        tpr_1 = (predictions[mask_1_pos] == 1).sum() / mask_1_pos.sum()
    else:
        tpr_1 = 0

    aeod = abs(tpr_0 - tpr_1)

    # ASPD (Statistical Parity Difference)
    mask_0 = (sensitive_attrs == 0)
    if mask_0.sum() > 0:
        pr_0 = (predictions[mask_0] == 1).sum() / mask_0.sum()
    else:
        pr_0 = 0

    mask_1 = (sensitive_attrs == 1)
    if mask_1.sum() > 0:
        pr_1 = (predictions[mask_1] == 1).sum() / mask_1.sum()
    else:
        pr_1 = 0

    aspd = abs(pr_0 - pr_1)

    return accuracy, aeod, aspd

# ============================================================================
# 主实验
# ============================================================================
def main():
    print("实验配置:")
    print(f"  数据集: {DATASET_NAME}")
    print(f"  客户端数量: {NUM_CLIENTS}")
    print(f"  恶意客户端: {NUM_MALICIOUS}")
    print(f"  训练轮次: {NUM_ROUNDS}")
    print(f"  批量大小: {BATCH_SIZE}")
    print(f"  学习率: {LEARNING_RATE}\n")

    # 加载数据
    print("1. 加载数据...")
    from src.data_loader import DatasetLoader

    data_loader = DatasetLoader(dataset_name=DATASET_NAME, seed=SEED, device=DEVICE)
    data_info = data_loader.get_info()

    print(f"   数据加载成功: {data_info['num_train']} 训练样本, {data_info['num_test']} 测试样本\n")

    # 获取数据
    tensors = data_loader.get_tensors()
    X_train = tensors['X_train']
    y_train = tensors['y_train']

    # 创建测试加载器
    test_loader = data_loader.create_test_loader(batch_size=BATCH_SIZE)

    # 数据分区
    print("2. 数据分区...")
    num_train = len(X_train)
    indices = np.arange(num_train)
    np.random.shuffle(indices)

    # 简单均匀分配
    client_data_size = num_train // NUM_CLIENTS
    client_data_indices = [indices[i*client_data_size:(i+1)*client_data_size]
                          for i in range(NUM_CLIENTS)]

    print(f"   每个客户端: ~{client_data_size} 样本\n")

    # 初始化模型
    print("3. 初始化模型...")
    global_model = SimpleMLP(input_size=data_info['num_features']).to(DEVICE)
    print(f"   模型参数: {sum(p.numel() for p in global_model.parameters())}\n")

    # 训练
    print(f"4. 开始训练 ({NUM_ROUNDS} 轮)...")

    for round_idx in range(NUM_ROUNDS):
        # 客户端本地训练
        client_updates = []

        for i, idx in enumerate(client_data_indices):
            # 本地模型
            local_model = copy.deepcopy(global_model)
            optimizer = torch.optim.SGD(local_model.parameters(), lr=LEARNING_RATE)
            criterion = nn.CrossEntropyLoss()

            # 本地数据
            X_local = X_train[idx]
            y_local = y_train[idx]

            # 训练
            local_model.train()
            dataset = torch.utils.data.TensorDataset(X_local, y_local)
            loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                outputs = local_model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

            # 计算更新
            update = {}
            for name, param in local_model.named_parameters():
                update[name] = param.data - global_model.state_dict()[name]
            client_updates.append(update)

        # 聚合
        aggregated_update = {}
        for name in global_model.state_dict():
            aggregated_update[name] = torch.stack([u[name] for u in client_updates]).mean(dim=0)

        # 更新全局模型
        new_state = global_model.state_dict()
        for name in new_state:
            new_state[name] = new_state[name] + aggregated_update[name]
        global_model.load_state_dict(new_state)

        # 评估
        if (round_idx + 1) % 2 == 0 or round_idx == NUM_ROUNDS - 1:
            acc, aeod, aspd = evaluate_model(global_model, test_loader, DEVICE)
            print(f"   轮次 {round_idx+1}/{NUM_ROUNDS}: "
                  f"ACC={acc:.4f}, AEOD={aeod:.4f}, ASPD={aspd:.4f}")

    # 最终评估
    print("\n5. 最终评估...")
    acc, aeod, aspd = evaluate_model(global_model, test_loader, DEVICE)

    print(f"\n最终结果:")
    print(f"  准确率 (ACC): {acc:.4f}")
    print(f"  公平性 (AEOD): {aeod:.4f}")
    print(f"  公平性 (ASPD): {aspd:.4f}")

    print("\n" + "="*80)
    print("实验完成！")
    print("="*80)

    print(f"""
与论文表4的对比 (FedAvg, IID, Benign):
  论文结果: ACC=83.05%, AEOD=0.018, ASPD=0.104
  当前结果: ACC={acc*100:.2f}%, AEOD={aeod:.3f}, ASPD={aspd:.3f}

注意:
1. 当前使用简化实现和较少训练轮次 ({NUM_ROUNDS}轮)
2. 论文使用50-100轮训练和Reweighting技术
3. 要获得更接近的结果，需要:
   - 增加训练轮次
   - 实现Reweighting
   - 使用Dirichlet分区
   - 调整超参数
""")

    return acc, aeod, aspd

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
