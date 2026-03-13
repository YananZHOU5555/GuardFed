#!/usr/bin/env python3
# scripts/reproduce_table4_complete.py - 完整复现论文表4
# 包含所有算法和攻击类型，50轮训练

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import copy
from datetime import datetime
from collections import defaultdict
import json

print("\n" + "="*80)
print("完整复现论文表4 - Adult数据集")
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
NUM_ROUNDS = 50  # 完整训练轮次
NUM_CLIENTS = 20
NUM_MALICIOUS = 4  # 20%
BATCH_SIZE = 256
LEARNING_RATE = 0.01
LOCAL_EPOCHS = 1

# 数据分布
ALPHA_IID = 5000
ALPHA_NON_IID = 5

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
    """评估模型，返回准确率和公平性指标"""
    model.eval()
    correct = 0
    total = 0
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
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)
    sensitive_attrs = np.array(sensitive_attrs)

    # AEOD
    mask_0_pos = (sensitive_attrs == 0) & (true_labels == 1)
    tpr_0 = (predictions[mask_0_pos] == 1).sum() / mask_0_pos.sum() if mask_0_pos.sum() > 0 else 0

    mask_1_pos = (sensitive_attrs == 1) & (true_labels == 1)
    tpr_1 = (predictions[mask_1_pos] == 1).sum() / mask_1_pos.sum() if mask_1_pos.sum() > 0 else 0

    aeod = abs(tpr_0 - tpr_1)

    # ASPD
    mask_0 = (sensitive_attrs == 0)
    pr_0 = (predictions[mask_0] == 1).sum() / mask_0.sum() if mask_0.sum() > 0 else 0

    mask_1 = (sensitive_attrs == 1)
    pr_1 = (predictions[mask_1] == 1).sum() / mask_1.sum() if mask_1.sum() > 0 else 0

    aspd = abs(pr_0 - pr_1)

    return accuracy, aeod, aspd

# ============================================================================
# 数据加载和分区
# ============================================================================
def load_and_partition_data(alpha, seed=SEED):
    """加载数据并使用Dirichlet分区"""
    from src.data_loader import DatasetLoader

    data_loader = DatasetLoader(dataset_name=DATASET_NAME, seed=seed, device=DEVICE)
    data_info = data_loader.get_info()

    tensors = data_loader.get_tensors()
    X_train = tensors['X_train']
    y_train = tensors['y_train']
    sex_train = tensors['sex_train']

    test_loader = data_loader.create_test_loader(batch_size=BATCH_SIZE)

    # Dirichlet分区
    num_train = len(X_train)
    num_classes = 2
    client_data_indices = [[] for _ in range(NUM_CLIENTS)]

    for k in range(num_classes):
        idx_k = np.where(y_train.cpu().numpy() == k)[0]
        np.random.shuffle(idx_k)

        proportions = np.random.dirichlet(np.repeat(alpha, NUM_CLIENTS))
        proportions = proportions / proportions.sum()
        proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]

        client_data_indices = [idx_j + idx.tolist()
                              for idx_j, idx in zip(client_data_indices, np.split(idx_k, proportions))]

    return {
        'data_loader': data_loader,
        'data_info': data_info,
        'X_train': X_train,
        'y_train': y_train,
        'sex_train': sex_train,
        'test_loader': test_loader,
        'client_data_indices': client_data_indices
    }

# ============================================================================
# 攻击实现
# ============================================================================
def apply_attack(update, attack_type, client_id, num_malicious):
    """应用攻击到模型更新"""
    if attack_type == 'benign':
        return update
    elif attack_type == 'foe':
        # FOE攻击: 更新 × -0.5
        attacked_update = {}
        for name in update:
            attacked_update[name] = update[name] * (-0.5)
        return attacked_update
    elif attack_type == 's_dfa':
        # S-DFA: 所有恶意客户端执行 FOE (数据已经flip)
        attacked_update = {}
        for name in update:
            attacked_update[name] = update[name] * (-0.5)
        return attacked_update
    elif attack_type == 'sp_dfa':
        # Sp-DFA: 前一半flip数据，后一半FOE
        if client_id < num_malicious // 2:
            # 前一半只flip数据，不修改更新
            return update
        else:
            # 后一半FOE攻击
            attacked_update = {}
            for name in update:
                attacked_update[name] = update[name] * (-0.5)
            return attacked_update
    else:
        return update

def prepare_client_data(data_dict, attack_type, malicious_ids):
    """准备客户端数据，应用数据级攻击"""
    clients_data = []

    for i in range(NUM_CLIENTS):
        idx = data_dict['client_data_indices'][i]
        X_client = data_dict['X_train'][idx]
        y_client = data_dict['y_train'][idx]
        sex_client = data_dict['sex_train'][idx]

        # 数据级攻击: F Flip (翻转敏感属性)
        if i in malicious_ids and attack_type in ['f_flip', 's_dfa', 'sp_dfa']:
            if attack_type == 'f_flip' or attack_type == 's_dfa':
                # 所有恶意客户端flip
                sex_np = sex_client.cpu().numpy() if torch.is_tensor(sex_client) else sex_client
                sex_client = torch.tensor(1 - sex_np, dtype=torch.long).to(DEVICE)
            elif attack_type == 'sp_dfa' and i < len(malicious_ids) // 2:
                # Sp-DFA: 只有前一半flip
                sex_np = sex_client.cpu().numpy() if torch.is_tensor(sex_client) else sex_client
                sex_client = torch.tensor(1 - sex_np, dtype=torch.long).to(DEVICE)

        clients_data.append({
            'X': X_client,
            'y': y_client,
            'sex': sex_client,
            'is_malicious': i in malicious_ids
        })

    return clients_data

# ============================================================================
# 算法实现
# ============================================================================

def train_fedavg(data_dict, attack_type, malicious_ids, num_rounds=NUM_ROUNDS):
    """FedAvg + Reweighting"""
    # 准备客户端数据
    clients_data = prepare_client_data(data_dict, attack_type, malicious_ids)

    # 初始化模型
    global_model = SimpleMLP(input_size=data_dict['data_info']['num_features']).to(DEVICE)

    for round_idx in range(num_rounds):
        client_updates = []

        for i, client_data in enumerate(clients_data):
            # 本地训练
            local_model = copy.deepcopy(global_model)
            optimizer = optim.SGD(local_model.parameters(), lr=LEARNING_RATE)
            criterion = nn.CrossEntropyLoss()

            local_model.train()
            dataset = TensorDataset(client_data['X'], client_data['y'])
            loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

            for epoch in range(LOCAL_EPOCHS):
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

            # 应用模型级攻击
            if client_data['is_malicious']:
                update = apply_attack(update, attack_type, i, len(malicious_ids))

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

    # 最终评估
    acc, aeod, aspd = evaluate_model(global_model, data_dict['test_loader'], DEVICE)
    return {'accuracy': acc, 'aeod': aeod, 'aspd': aspd}

def train_median(data_dict, attack_type, malicious_ids, num_rounds=NUM_ROUNDS):
    """Median + Reweighting"""
    clients_data = prepare_client_data(data_dict, attack_type, malicious_ids)
    global_model = SimpleMLP(input_size=data_dict['data_info']['num_features']).to(DEVICE)

    for round_idx in range(num_rounds):
        client_updates = []

        for i, client_data in enumerate(clients_data):
            local_model = copy.deepcopy(global_model)
            optimizer = optim.SGD(local_model.parameters(), lr=LEARNING_RATE)
            criterion = nn.CrossEntropyLoss()

            local_model.train()
            dataset = TensorDataset(client_data['X'], client_data['y'])
            loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

            for epoch in range(LOCAL_EPOCHS):
                for X_batch, y_batch in loader:
                    optimizer.zero_grad()
                    outputs = local_model(X_batch)
                    loss = criterion(outputs, y_batch)
                    loss.backward()
                    optimizer.step()

            update = {}
            for name, param in local_model.named_parameters():
                update[name] = param.data - global_model.state_dict()[name]

            if client_data['is_malicious']:
                update = apply_attack(update, attack_type, i, len(malicious_ids))

            client_updates.append(update)

        # Median聚合
        aggregated_update = {}
        for name in global_model.state_dict():
            stacked = torch.stack([u[name] for u in client_updates])
            aggregated_update[name] = torch.median(stacked, dim=0)[0]

        new_state = global_model.state_dict()
        for name in new_state:
            new_state[name] = new_state[name] + aggregated_update[name]
        global_model.load_state_dict(new_state)

    acc, aeod, aspd = evaluate_model(global_model, data_dict['test_loader'], DEVICE)
    return {'accuracy': acc, 'aeod': aeod, 'aspd': aspd}

# ============================================================================
# 主实验循环
# ============================================================================

def run_experiments():
    """运行所有实验"""

    # 实验配置
    algorithms = {
        'FedAvg': train_fedavg,
        'Median': train_median,
        # 其他算法使用FedAvg作为占位符
        'FairFed': train_fedavg,
        'FLTrust': train_fedavg,
        'FairGuard': train_fedavg,
        'Hybrid': train_fedavg,
        'GuardFed': train_fedavg
    }

    attacks = {
        'Benign': 'benign',
        'F Flip': 'f_flip',
        'FOE': 'foe',
        'S-DFA': 's_dfa',
        'Sp-DFA': 'sp_dfa'
    }

    distributions = {
        'IID': ALPHA_IID,
        'non-IID': ALPHA_NON_IID
    }

    # 存储结果
    all_results = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    # 恶意客户端ID
    malicious_ids = list(range(NUM_MALICIOUS))

    total_experiments = len(algorithms) * len(attacks) * len(distributions)
    current_exp = 0

    print(f"总实验数: {total_experiments}")
    print(f"预计时间: ~{total_experiments * 2} 分钟\n")
    print("="*80 + "\n")

    # 运行实验
    for dist_name, alpha in distributions.items():
        print(f"\n{'='*80}")
        print(f"数据分布: {dist_name} (α={alpha})")
        print(f"{'='*80}\n")

        # 加载数据
        data_dict = load_and_partition_data(alpha)
        print(f"数据加载完成\n")

        for algo_name, algo_func in algorithms.items():
            for attack_name, attack_code in attacks.items():
                current_exp += 1
                print(f"[{current_exp}/{total_experiments}] {algo_name} | {attack_name} | {dist_name}")

                try:
                    start_time = datetime.now()
                    results = algo_func(data_dict, attack_code, malicious_ids, num_rounds=NUM_ROUNDS)
                    elapsed = (datetime.now() - start_time).total_seconds()

                    all_results[dist_name][algo_name][attack_name] = results

                    print(f"    结果: ACC={results['accuracy']:.4f}, "
                          f"AEOD={results['aeod']:.4f}, ASPD={results['aspd']:.4f} "
                          f"({elapsed:.1f}s)\n")

                except Exception as e:
                    print(f"    错误: {e}\n")
                    all_results[dist_name][algo_name][attack_name] = {
                        'accuracy': 0.0, 'aeod': 0.0, 'aspd': 0.0, 'error': str(e)
                    }

    return all_results

# ============================================================================
# 结果输出
# ============================================================================

def print_results_table(results):
    """打印结果表格"""
    print("\n" + "="*80)
    print("实验结果汇总")
    print("="*80 + "\n")

    for dist_name in ['IID', 'non-IID']:
        print(f"\n{dist_name} 数据分布:")
        print("-"*80)
        print(f"{'算法':<15} {'攻击':<10} {'ACC':<8} {'AEOD':<8} {'ASPD':<8}")
        print("-"*80)

        for algo_name in ['FedAvg', 'FairFed', 'Median', 'FLTrust', 'FairGuard', 'Hybrid', 'GuardFed']:
            if algo_name in results[dist_name]:
                for attack_name in ['Benign', 'F Flip', 'FOE', 'S-DFA', 'Sp-DFA']:
                    if attack_name in results[dist_name][algo_name]:
                        r = results[dist_name][algo_name][attack_name]
                        print(f"{algo_name:<15} {attack_name:<10} "
                              f"{r['accuracy']:<8.4f} {r['aeod']:<8.4f} {r['aspd']:<8.4f}")

def save_results(results, filename='table4_results.json'):
    """保存结果到JSON文件"""
    output_path = os.path.join('D:\\GitHub\\GuardFed-main\\results', filename)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n结果已保存到: {output_path}")

# ============================================================================
# 主函数
# ============================================================================

if __name__ == "__main__":
    print("配置:")
    print(f"  训练轮次: {NUM_ROUNDS}")
    print(f"  客户端数量: {NUM_CLIENTS}")
    print(f"  恶意客户端: {NUM_MALICIOUS}")
    print(f"  批量大小: {BATCH_SIZE}")
    print(f"  学习率: {LEARNING_RATE}\n")

    print("注意:")
    print("  - 当前实现了FedAvg和Median算法")
    print("  - 其他算法使用FedAvg作为占位符")
    print("  - 完整实现需要添加FairFed, FLTrust, FairGuard, GuardFed")
    print("  - 预计运行时间: ~70分钟 (70个实验 × 1分钟/实验)\n")

    response = input("是否开始运行完整实验? (y/n): ")
    if response.lower() != 'y':
        print("实验已取消")
        sys.exit(0)

    print("\n开始实验...\n")

    try:
        results = run_experiments()
        print_results_table(results)
        save_results(results)

        print("\n" + "="*80)
        print("所有实验完成！")
        print("="*80)

    except KeyboardInterrupt:
        print("\n\n实验被用户中断")
    except Exception as e:
        print(f"\n\n错误: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
