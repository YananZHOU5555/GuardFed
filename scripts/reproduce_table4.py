#!/usr/bin/env python3
# scripts/reproduce_table4.py - 复现论文表4 (Adult数据集)
# 排除Class-B FL，包含其他所有方法

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import copy
from datetime import datetime
from collections import defaultdict

# 导入配置
from src.HYPERPARAMETERS import HYPERPARAMETERS, SEED, NUM_MALICIOUS_CLIENTS
from src.data_loader import DatasetLoader
from src.models.function import (
    compute_fairness_metrics,
    test_inference_modified,
    compute_reweighing_weights,
    MLP
)

print("\n" + "="*80)
print("复现论文表4 - Adult数据集实验")
print("="*80)
print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80 + "\n")

# ============================================================================
# 实验配置
# ============================================================================
DATASET_NAME = 'adult'
NUM_ROUNDS = 10  # 减少轮次以加快测试，完整实验使用50-100轮
NUM_CLIENTS = 20
NUM_MALICIOUS = 4  # 20%恶意客户端
BATCH_SIZE = 256
LEARNING_RATE = 0.01
SERVER_DATA_RATIO = 0.05  # 5%服务器数据（1%真实 + 4%合成）

# 数据分布设置
ALPHA_IID = 5000  # IID
ALPHA_NON_IID = 5  # non-IID

# 攻击类型
ATTACK_TYPES = {
    'Benign': 'no_attack',
    'F Flip': 'label_flip',
    'FOE': 'foe',
    'S-DFA': 's_dfa',
    'Sp-DFA': 'sp_dfa'
}

# 测试的算法（排除Class-B FL）
ALGORITHMS = [
    'FedAvg_RW',      # FedAvg + Reweighting
    'FairFed_RW',     # FairFed + Reweighting
    'Medium_RW',      # Median + Reweighting
    'FLTrust_RW',     # FLTrust + Reweighting
    'FedAvg_RW_FairG',      # FairGuard
    'FLTrust_RW_FairG',     # FLTrust + FairGuard (Hybrid)
    'FedAvg_RW_FairCosG'    # GuardFed (我们的方法)
]

# 算法显示名称映射
ALGORITHM_NAMES = {
    'FedAvg_RW': 'FedAvg',
    'FairFed_RW': 'FairFed',
    'Medium_RW': 'Median',
    'FLTrust_RW': 'FLTrust',
    'FedAvg_RW_FairG': 'FairGuard',
    'FLTrust_RW_FairG': 'FLTrust+FairGuard',
    'FedAvg_RW_FairCosG': 'GuardFed'
}

# ============================================================================
# 辅助函数
# ============================================================================

def load_and_partition_data(alpha, seed=SEED):
    """加载并分区数据"""
    print(f"\n加载数据 (α={alpha})...")

    # 加载数据
    data_loader = DatasetLoader(dataset_name=DATASET_NAME, seed=seed, device=HYPERPARAMETERS['DEVICE'])
    data_info = data_loader.get_info()

    # 获取张量数据
    tensors = data_loader.get_tensors()
    X_train_tensor = tensors['X_train']
    y_train_tensor = tensors['y_train']
    sex_train = tensors['sex_train']

    # 获取DataFrame数据
    train_df = data_loader.train_df

    # 服务器数据分割
    num_train = len(X_train_tensor)
    num_server = int(num_train * SERVER_DATA_RATIO)
    num_clients_data = num_train - num_server

    indices = np.arange(num_train)
    np.random.shuffle(indices)
    server_indices = indices[:num_server]
    client_indices = indices[num_server:]

    X_server = X_train_tensor[server_indices]
    y_server = y_train_tensor[server_indices]
    sex_server = sex_train[server_indices] if torch.is_tensor(sex_train) else torch.tensor(sex_train[server_indices])

    X_clients = X_train_tensor[client_indices]
    y_clients = y_train_tensor[client_indices]
    sex_clients = sex_train[client_indices] if torch.is_tensor(sex_train) else sex_train[client_indices]

    # Dirichlet分区
    num_classes = HYPERPARAMETERS['OUTPUT_SIZE']
    client_data_indices = [[] for _ in range(NUM_CLIENTS)]

    for k in range(num_classes):
        idx_k = np.where(y_clients.cpu().numpy() == k)[0]
        np.random.shuffle(idx_k)
        proportions = np.random.dirichlet(np.repeat(alpha, NUM_CLIENTS))
        proportions = np.array([p * (len(idx_j) < num_clients_data / NUM_CLIENTS)
                               for p, idx_j in zip(proportions, client_data_indices)])
        proportions = proportions / proportions.sum()
        proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
        client_data_indices = [idx_j + idx.tolist()
                              for idx_j, idx in zip(client_data_indices, np.split(idx_k, proportions))]

    # 计算Reweighting权重
    reweighing_weights = compute_reweighing_weights(
        train_df,
        data_loader.sensitive_column,
        'income'
    )

    print(f"   数据加载完成: {data_info['num_train']} 训练样本, {data_info['num_test']} 测试样本")
    print(f"   服务器数据: {len(X_server)} 样本")
    print(f"   客户端数据: {len(X_clients)} 样本\n")

    return {
        'data_loader': data_loader,
        'data_info': data_info,
        'X_server': X_server,
        'y_server': y_server,
        'sex_server': sex_server,
        'X_clients': X_clients,
        'y_clients': y_clients,
        'sex_clients': sex_clients,
        'client_data_indices': client_data_indices,
        'reweighing_weights': reweighing_weights,
        'train_df': train_df
    }

def create_clients_simple(data_dict, attack_type, malicious_ids):
    """创建简化的客户端数据结构"""
    clients_data = []

    for i in range(NUM_CLIENTS):
        idx = data_dict['client_data_indices'][i]
        X_client = data_dict['X_clients'][idx]
        y_client = data_dict['y_clients'][idx]
        sex_client = data_dict['sex_clients'][idx]

        # 确定攻击类型
        if i in malicious_ids:
            if attack_type == 'label_flip':
                # 翻转敏感属性
                sex_client_np = sex_client.cpu().numpy() if torch.is_tensor(sex_client) else sex_client
                sex_client_flipped = 1 - sex_client_np
                sex_client = torch.tensor(sex_client_flipped, dtype=torch.long).to(HYPERPARAMETERS['DEVICE'])
            elif attack_type in ['foe', 's_dfa', 'sp_dfa']:
                # 这些攻击在训练后处理
                pass

        # 计算样本权重
        sex_np = sex_client.cpu().numpy() if torch.is_tensor(sex_client) else sex_client
        y_np = y_client.cpu().numpy() if torch.is_tensor(y_client) else y_client
        weights = np.array([data_dict['reweighing_weights'].get((s, c), 1.0)
                           for s, c in zip(sex_np, y_np)])

        clients_data.append({
            'X': X_client,
            'y': y_client,
            'sex': sex_client,
            'weights': weights,
            'is_malicious': i in malicious_ids,
            'attack_type': attack_type if i in malicious_ids else 'no_attack'
        })

    return clients_data

def train_fedavg(data_dict, clients_data, num_rounds=NUM_ROUNDS):
    """训练FedAvg + Reweighting"""
    # 初始化全局模型
    global_model = MLP(
        num_features=data_dict['data_info']['num_features'],
        num_classes=2,
        seed=SEED
    ).to(HYPERPARAMETERS['DEVICE'])

    test_loader = data_dict['data_loader'].create_test_loader(batch_size=BATCH_SIZE)

    for round_idx in range(num_rounds):
        # 客户端本地训练
        client_updates = []

        for client_data in clients_data:
            # 创建本地模型
            local_model = copy.deepcopy(global_model)
            optimizer = torch.optim.SGD(local_model.parameters(), lr=LEARNING_RATE)
            criterion = torch.nn.CrossEntropyLoss(reduction='none')

            # 本地训练
            local_model.train()
            dataset = torch.utils.data.TensorDataset(
                client_data['X'],
                client_data['y'],
                torch.tensor(client_data['weights'], dtype=torch.float32).to(HYPERPARAMETERS['DEVICE'])
            )
            loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

            for X_batch, y_batch, w_batch in loader:
                optimizer.zero_grad()
                outputs = local_model(X_batch)
                loss = criterion(outputs, y_batch)
                weighted_loss = (loss * w_batch).mean()
                weighted_loss.backward()
                optimizer.step()

            # 计算更新
            update = {}
            for name, param in local_model.named_parameters():
                update[name] = param.data - global_model.state_dict()[name]

            # 应用攻击
            if client_data['is_malicious']:
                if client_data['attack_type'] == 'foe':
                    # FOE攻击: 更新 × -0.5
                    for name in update:
                        update[name] = update[name] * (-0.5)
                elif client_data['attack_type'] == 's_dfa':
                    # S-DFA: label_flip + FOE
                    for name in update:
                        update[name] = update[name] * (-0.5)

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
    loss, acc, fairness_metrics, _ = test_inference_modified(
        global_model=global_model,
        test_loader=test_loader,
        model_class=MLP
    )

    return {
        'accuracy': acc,
        'aeod': fairness_metrics['EOD'],
        'aspd': fairness_metrics['SPD']
    }

def run_single_experiment(algorithm, attack_type, alpha, seed=SEED):
    """运行单个实验"""
    print(f"   运行: {ALGORITHM_NAMES[algorithm]} | {attack_type} | α={alpha}")

    # 加载数据
    data_dict = load_and_partition_data(alpha, seed)

    # 确定恶意客户端
    malicious_ids = list(range(NUM_MALICIOUS))

    # 创建客户端
    attack_code = ATTACK_TYPES[attack_type]
    clients_data = create_clients_simple(data_dict, attack_code, malicious_ids)

    # 训练（这里简化为只实现FedAvg，其他算法类似）
    if algorithm == 'FedAvg_RW':
        results = train_fedavg(data_dict, clients_data, num_rounds=NUM_ROUNDS)
    else:
        # 其他算法的实现类似，这里为了快速测试，暂时返回模拟结果
        print(f"      注意: {algorithm} 使用简化实现")
        results = train_fedavg(data_dict, clients_data, num_rounds=NUM_ROUNDS)

    return results

# ============================================================================
# 主实验循环
# ============================================================================

print("开始实验...\n")

# 存储结果
all_results = defaultdict(lambda: defaultdict(dict))

# 测试配置
test_configs = [
    ('IID', ALPHA_IID),
    ('non-IID', ALPHA_NON_IID)
]

# 只测试一个算法和一个攻击来验证流程
print("="*80)
print("快速验证测试 (仅测试FedAvg + Benign)")
print("="*80 + "\n")

for dist_name, alpha in test_configs:
    print(f"\n{'='*80}")
    print(f"数据分布: {dist_name} (α={alpha})")
    print(f"{'='*80}\n")

    # 只测试一个攻击类型
    attack_type = 'Benign'
    algorithm = 'FedAvg_RW'

    try:
        results = run_single_experiment(algorithm, attack_type, alpha)
        all_results[dist_name][algorithm][attack_type] = results

        print(f"      结果: ACC={results['accuracy']:.4f}, "
              f"AEOD={results['aeod']:.4f}, ASPD={results['aspd']:.4f}")
    except Exception as e:
        print(f"      错误: {e}")
        import traceback
        traceback.print_exc()

# ============================================================================
# 输出结果
# ============================================================================

print("\n" + "="*80)
print("实验完成！")
print("="*80 + "\n")

print("结果摘要:")
print("-"*80)
for dist_name in ['IID', 'non-IID']:
    print(f"\n{dist_name}:")
    for algorithm in ['FedAvg_RW']:
        if algorithm in all_results[dist_name]:
            print(f"  {ALGORITHM_NAMES[algorithm]}:")
            for attack_type in ['Benign']:
                if attack_type in all_results[dist_name][algorithm]:
                    r = all_results[dist_name][algorithm][attack_type]
                    print(f"    {attack_type}: ACC={r['accuracy']:.4f}, "
                          f"AEOD={r['aeod']:.4f}, ASPD={r['aspd']:.4f}")

print("\n" + "="*80)
print("注意: 这是快速验证版本，只测试了FedAvg+Benign")
print("完整实验需要实现所有算法和攻击类型")
print("="*80)
