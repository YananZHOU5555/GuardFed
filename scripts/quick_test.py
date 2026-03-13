#!/usr/bin/env python3
# scripts/quick_test.py - 快速测试脚本，用于验证代码完整性

import sys
import os
# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import copy
from datetime import datetime

# 导入快速测试配置
from src.HYPERPARAMETERS_QUICK_TEST import HYPERPARAMETERS, SEED, NUM_MALICIOUS_CLIENTS
from src.data_loader import DatasetLoader
from src.models.function import (
    compute_fairness_metrics,
    test_inference_modified,
    compute_reweighing_weights,
    assign_sample_weights_to_clients,
    MLP
)

# 导入算法
from src.algorithms.FedAvg import Server as FedAvgServer, Client as FedAvgClient
from src.algorithms.FairCosG import FairCosG

print("\n" + "="*80)
print("GuardFed 快速测试 - 代码完整性验证")
print("="*80)
print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"训练轮次: {HYPERPARAMETERS['NUM_GLOBAL_ROUNDS']}")
print(f"客户端数量: {HYPERPARAMETERS['NUM_CLIENTS']}")
from src.HYPERPARAMETERS_QUICK_TEST import NUM_MALICIOUS_CLIENTS
print(f"恶意客户端数量: {NUM_MALICIOUS_CLIENTS}")
print("="*80 + "\n")

# ============================================================================
# 1. 加载数据集
# ============================================================================
DATASET_NAME = 'adult'
print(f"[1/5] 加载数据集: {DATASET_NAME.upper()}")

data_loader = DatasetLoader(dataset_name=DATASET_NAME, seed=SEED, device=HYPERPARAMETERS['DEVICE'])
data_info = data_loader.get_info()

# 获取数据
tensors = data_loader.get_tensors()
X_train_tensor = tensors['X_train']
y_train_tensor = tensors['y_train']
X_test_tensor = tensors['X_test']
y_test_tensor = tensors['y_test']
sex_train = tensors['sex_train']
sex_test_tensor = tensors['sex_test']

# 获取DataFrame格式数据
train_df = data_loader.train_df
test_df = data_loader.test_df
X_train = data_loader.X_train
y_train = data_loader.y_train
X_test = data_loader.X_test
y_test = data_loader.y_test
sex_test = data_loader.sex_test

# 敏感属性配置
SENSITIVE_COLUMN = data_loader.sensitive_column
A_PRIVILEGED = data_loader.A_PRIVILEGED
A_UNPRIVILEGED = data_loader.A_UNPRIVILEGED

# 创建测试数据加载器
test_loader = data_loader.create_test_loader(batch_size=HYPERPARAMETERS['BATCH_SIZE'])

# 设置输入特征维度
HYPERPARAMETERS['INPUT_SIZE'] = data_info['num_features']
print(f"   输入特征维度: {HYPERPARAMETERS['INPUT_SIZE']}")
print(f"   训练样本数: {data_info['num_train']}")
print(f"   测试样本数: {data_info['num_test']}")
print(f"   敏感属性: {SENSITIVE_COLUMN}\n")

# ============================================================================
# 2. 数据分区
# ============================================================================
print(f"[2/5] 数据分区 (Dirichlet α=1, Non-IID)")

# 服务器数据分割
server_data_ratio = HYPERPARAMETERS['SERVER_DATA_RATIO']
num_train = len(X_train_tensor)
num_server = int(num_train * server_data_ratio)
num_clients_data = num_train - num_server

indices = np.arange(num_train)
np.random.shuffle(indices)
server_indices = indices[:num_server]
client_indices = indices[num_server:]

X_server = X_train_tensor[server_indices]
y_server = y_train_tensor[server_indices]
sex_server = sex_train[server_indices]

X_clients = X_train_tensor[client_indices]
y_clients = y_train_tensor[client_indices]
sex_clients = sex_train[client_indices]

# Dirichlet分区
num_clients = HYPERPARAMETERS['NUM_CLIENTS']
alpha = HYPERPARAMETERS['ALPHA_DIRICHLET'][0]
num_classes = HYPERPARAMETERS['OUTPUT_SIZE']

client_data_indices = [[] for _ in range(num_clients)]
for k in range(num_classes):
    idx_k = np.where(y_clients.cpu().numpy() == k)[0]
    np.random.shuffle(idx_k)
    proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
    proportions = np.array([p * (len(idx_j) < num_clients_data / num_clients) for p, idx_j in zip(proportions, client_data_indices)])
    proportions = proportions / proportions.sum()
    proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
    client_data_indices = [idx_j + idx.tolist() for idx_j, idx in zip(client_data_indices, np.split(idx_k, proportions))]

print(f"   服务器数据: {len(X_server)} 样本")
print(f"   客户端总数据: {len(X_clients)} 样本")
print(f"   每个客户端平均: {len(X_clients)//num_clients} 样本\n")

# ============================================================================
# 3. 计算Reweighting权重
# ============================================================================
print(f"[3/5] 计算Reweighting权重")

reweighing_weights = compute_reweighing_weights(
    train_df,
    SENSITIVE_COLUMN,
    'income'  # class column name
)

client_sample_weights = []
for i in range(num_clients):
    idx = client_data_indices[i]
    client_sex = sex_clients[idx].cpu().numpy() if torch.is_tensor(sex_clients[idx]) else sex_clients[idx]
    client_y = y_clients[idx].cpu().numpy() if torch.is_tensor(y_clients[idx]) else y_clients[idx]
    weights = np.array([reweighing_weights.get((s, c), 1.0) for s, c in zip(client_sex, client_y)])
    client_sample_weights.append(weights)

print(f"   Reweighting权重计算完成\n")

# ============================================================================
# 4. 初始化客户端
# ============================================================================
print(f"[4/5] 初始化客户端")

num_malicious = NUM_MALICIOUS_CLIENTS
malicious_clients = list(range(num_malicious))

clients = []
for i in range(num_clients):
    idx = client_data_indices[i]
    X_client = X_clients[idx]
    y_client = y_clients[idx]
    sex_client = sex_clients[idx]
    sample_weights = client_sample_weights[i]

    # 创建数据字典
    client_data = {
        "X": X_client,
        "y": y_client,
        "sensitive": sex_client.cpu().numpy() if torch.is_tensor(sex_client) else sex_client,
        "sample_weights": sample_weights
    }

    # 设置攻击类型
    if i < num_malicious:
        attack_form = 'attack_fair_1'  # label_flip攻击
    else:
        attack_form = 'no_attack'

    client = FedAvgClient(
        client_id=i,
        data=client_data,
        sensitive_features=client_data["sensitive"],
        batch_size=HYPERPARAMETERS['BATCH_SIZE'],
        learning_rate=HYPERPARAMETERS['LEARNING_RATES']['FedAvg_RW'][0],
        model_class=MLP,
        input_size=HYPERPARAMETERS['INPUT_SIZE'],
        attack_form=attack_form
    )
    clients.append(client)

print(f"   总客户端: {num_clients}")
print(f"   恶意客户端: {num_malicious} (ID: {malicious_clients})")
print(f"   良性客户端: {num_clients - num_malicious}\n")

# ============================================================================
# 5. 训练 (FedAvg_RW_FairCosG)
# ============================================================================
print(f"[5/5] 开始训练 GuardFed (FedAvg_RW_FairCosG)")
print("-"*80)

# 初始化全局模型
global_model = MLP(
    input_size=HYPERPARAMETERS['INPUT_SIZE'],
    output_size=HYPERPARAMETERS['OUTPUT_SIZE']
).to(HYPERPARAMETERS['DEVICE'])

# 初始化FairCosG防御
faircosg = FairCosG(
    server_model=copy.deepcopy(global_model),
    X_server=X_server,
    y_server=y_server,
    sex_server=sex_server,
    batch_size=HYPERPARAMETERS['BATCH_SIZE'],
    learning_rate=HYPERPARAMETERS['FAIRCOSG_BETA'],
    num_epochs=HYPERPARAMETERS['FAIRCOSG_SERVER_EPOCHS'],
    device=HYPERPARAMETERS['DEVICE'],
    lambda_fairness=HYPERPARAMETERS['FAIRCOSG_LAMBDA_VALUES'][0],
    eod_tolerance=HYPERPARAMETERS['FAIRCOSG_EOD_TOLERANCE'],
    score_threshold=HYPERPARAMETERS['FAIRCOSG_SCORE_THRESHOLD'],
    A_PRIVILEGED=A_PRIVILEGED,
    A_UNPRIVILEGED=A_UNPRIVILEGED
)

# 训练循环
num_rounds = HYPERPARAMETERS['NUM_GLOBAL_ROUNDS']
results = []

for round_idx in range(num_rounds):
    print(f"\n轮次 {round_idx + 1}/{num_rounds}")

    # 客户端本地训练
    client_models = []
    for client in clients:
        client.set_model(copy.deepcopy(global_model))
        client.train()
        client_models.append(client.get_model())

    # FairCosG聚合
    global_model = faircosg.aggregate(
        global_model=global_model,
        client_models=client_models,
        round_idx=round_idx
    )

    # 评估
    acc, eod, spd, _, _, _ = test_inference_modified(
        model=global_model,
        test_loader=test_loader,
        sex_test=sex_test_tensor,
        device=HYPERPARAMETERS['DEVICE'],
        A_PRIVILEGED=A_PRIVILEGED,
        A_UNPRIVILEGED=A_UNPRIVILEGED
    )

    print(f"   准确率: {acc:.4f} | EOD: {eod:.4f} | SPD: {spd:.4f}")

    results.append({
        'round': round_idx + 1,
        'accuracy': acc,
        'eod': eod,
        'spd': spd
    })

# ============================================================================
# 6. 输出结果
# ============================================================================
print("\n" + "="*80)
print("测试完成！")
print("="*80)
print("\n最终结果:")
print(f"  准确率: {results[-1]['accuracy']:.4f}")
print(f"  EOD: {results[-1]['eod']:.4f}")
print(f"  SPD: {results[-1]['spd']:.4f}")

print("\n所有轮次结果:")
print("-"*80)
print(f"{'轮次':<10} {'准确率':<15} {'EOD':<15} {'SPD':<15}")
print("-"*80)
for r in results:
    print(f"{r['round']:<10} {r['accuracy']:<15.4f} {r['eod']:<15.4f} {r['spd']:<15.4f}")

print("\n" + "="*80)
print("代码完整性验证通过！所有模块正常工作。")
print("="*80)
