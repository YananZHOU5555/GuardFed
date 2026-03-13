#!/usr/bin/env python3
# 快速测试 - 只运行2个实验
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import copy
from datetime import datetime
from collections import defaultdict

# 配置
NUM_ROUNDS = 3  # 只用3轮快速测试
NUM_CLIENTS = 20
NUM_MALICIOUS = 4
ALPHA_IID = 5000

# 设置全局变量 - 必须在导入算法之前
import src.HYPERPARAMETERS as HP_module
import src.algorithms.FedAvg as FedAvg_module
import src.algorithms.FLTrust as FLTrust_module
import src.algorithms.FairFed as FairFed_module
import src.algorithms.Medium as Medium_module

HP_module.MALICIOUS_CLIENTS = list(range(NUM_MALICIOUS))
FedAvg_module.MALICIOUS_CLIENTS = list(range(NUM_MALICIOUS))
FLTrust_module.MALICIOUS_CLIENTS = list(range(NUM_MALICIOUS))
FairFed_module.MALICIOUS_CLIENTS = list(range(NUM_MALICIOUS))
Medium_module.MALICIOUS_CLIENTS = list(range(NUM_MALICIOUS))

# 导入配置和模块
from src.HYPERPARAMETERS import HYPERPARAMETERS, SEED

# 设置HYPERPARAMETERS到各个模块（包括function模块）
import src.models.function as function_module
function_module.HYPERPARAMETERS = HYPERPARAMETERS
FedAvg_module.HYPERPARAMETERS = HYPERPARAMETERS
FLTrust_module.HYPERPARAMETERS = HYPERPARAMETERS
FairFed_module.HYPERPARAMETERS = HYPERPARAMETERS
Medium_module.HYPERPARAMETERS = HYPERPARAMETERS
FedAvg_module.SEED = SEED
FLTrust_module.SEED = SEED
FairFed_module.SEED = SEED
Medium_module.SEED = SEED
FedAvg_module.DEVICE = HYPERPARAMETERS['DEVICE']
FLTrust_module.DEVICE = HYPERPARAMETERS['DEVICE']
FairFed_module.DEVICE = HYPERPARAMETERS['DEVICE']
Medium_module.DEVICE = HYPERPARAMETERS['DEVICE']

# 导入函数并设置到模块
from src.models.function import test_inference_modified as test_func
FedAvg_module.test_inference_modified = test_func
FLTrust_module.test_inference_modified = test_func
FairFed_module.test_inference_modified = test_func
Medium_module.test_inference_modified = test_func
FedAvg_module.DEVICE = HYPERPARAMETERS['DEVICE']
FLTrust_module.DEVICE = HYPERPARAMETERS['DEVICE']
FairFed_module.DEVICE = HYPERPARAMETERS['DEVICE']
Medium_module.DEVICE = HYPERPARAMETERS['DEVICE']

from src.data_loader import DatasetLoader
from src.models.function import compute_reweighing_weights, MLP, test_inference_modified
from src.algorithms.FedAvg import Server as FedAvgServer, Client as FedAvgClient
from src.algorithms.FLTrust import Server as FLTrustServer, Client as FLTrustClient

print("\n" + "="*80)
print("快速测试 - 检测错误")
print("="*80)
print(f"配置: {NUM_ROUNDS}轮, {NUM_CLIENTS}客户端, {NUM_MALICIOUS}恶意\n")

def run_single_experiment(algorithm, attack, alpha):
    """运行单个实验"""
    print(f"测试: {algorithm} | {attack}")

    # 加载数据
    data_loader = DatasetLoader(dataset_name='adult', seed=SEED, device=HYPERPARAMETERS['DEVICE'])
    data_info = data_loader.get_info()

    # 设置全局变量（FLTrust需要）
    FLTrust_module.scaler = data_loader.scaler
    FLTrust_module.numerical_columns = data_loader.numerical_columns
    
    # 获取数据
    tensors = data_loader.get_tensors()
    X_train, y_train, sex_train = tensors['X_train'], tensors['y_train'], tensors['sex_train']
    test_loader = data_loader.create_test_loader(batch_size=HYPERPARAMETERS['BATCH_SIZE'])
    
    # Dirichlet分区
    num_train, num_classes = len(X_train), 2
    client_data_indices = [[] for _ in range(NUM_CLIENTS)]
    for k in range(num_classes):
        idx_k = np.where(y_train.cpu().numpy() == k)[0]
        np.random.shuffle(idx_k)
        proportions = np.random.dirichlet(np.repeat(alpha, NUM_CLIENTS))
        proportions = proportions / proportions.sum()
        proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
        client_data_indices = [idx_j + idx.tolist() for idx_j, idx in zip(client_data_indices, np.split(idx_k, proportions))]
    
    # 计算Reweighting权重
    reweighing_weights = compute_reweighing_weights(data_loader.train_df, data_loader.sensitive_column, 'income')
    
    # 创建客户端
    clients = []
    for i in range(NUM_CLIENTS):
        idx = client_data_indices[i]
        X_client, y_client = X_train[idx], y_train[idx]
        sex_client = sex_train[idx] if torch.is_tensor(sex_train) else sex_train[idx]
        
        # 计算样本权重
        sex_np = sex_client.cpu().numpy() if torch.is_tensor(sex_client) else sex_client
        y_np = y_client.cpu().numpy()
        weights = np.array([reweighing_weights.get((s, c), 1.0) for s, c in zip(sex_np, y_np)])
        
        # 攻击映射
        attack_map = {'benign': 'no_attack', 'f_flip': 'attack_fair_1', 'foe': 'attack_acc_0.5'}
        attack_form = attack_map.get(attack, 'no_attack')
        
        client_data = {'X': X_client, 'y': y_client, 'sensitive': sex_np, 'sample_weights': weights}
        
        if 'FLTrust' in algorithm:
            client = FLTrustClient(i, client_data, sex_np, HYPERPARAMETERS['BATCH_SIZE'], 
                                  HYPERPARAMETERS['LEARNING_RATES']['FLTrust_RW'][0], MLP, 
                                  data_info['num_features'], attack_form)
        else:
            client = FedAvgClient(i, client_data, sex_np, HYPERPARAMETERS['BATCH_SIZE'],
                                 HYPERPARAMETERS['LEARNING_RATES']['FedAvg_RW'][0], MLP,
                                 data_info['num_features'], attack_form)
        clients.append(client)
    
    # 初始化全局模型
    global_model = MLP(num_features=data_info['num_features'], num_classes=2, seed=SEED).to(HYPERPARAMETERS['DEVICE'])

    # 准备服务器数据（FLTrust需要）
    server_data = None
    if 'FLTrust' in algorithm:
        # 使用10%的训练数据作为服务器数据
        num_server = int(len(data_loader.train_df) * 0.1)
        server_data = data_loader.train_df.sample(n=num_server, random_state=SEED)

    # 创建服务器
    if 'FLTrust' in algorithm:
        server = FLTrustServer(global_model, clients, algorithm, HYPERPARAMETERS, server_data=server_data)
    else:
        server = FedAvgServer(global_model, clients, algorithm, HYPERPARAMETERS)

    # 设置test_loader为全局变量
    FedAvg_module.test_loader = test_loader
    FLTrust_module.test_loader = test_loader

    # 训练
    for round_idx in range(NUM_ROUNDS):
        print(f"\n=== Round {round_idx + 1}/{NUM_ROUNDS} ===")
        # 在训练前检查模型准确率和权重
        if round_idx == 0:
            loss_before, acc_before, fm_before, _ = test_inference_modified(global_model, test_loader, MLP)
            print(f"训练前: ACC={acc_before:.4f}, AEOD={fm_before['EOD']:.4f}, ASPD={fm_before['SPD']:.4f}")
            # 打印第一层权重的一些值
            first_weight = list(global_model.state_dict().values())[0]
            print(f"初始权重样本: {first_weight[0, :5]}")

        server.run_round(round_idx, None, None, MLP)

        # 同步Server的global_model到外部global_model
        global_model.load_state_dict(server.global_model.state_dict())

        # 训练后检查
        loss_after, acc_after, fm_after, _ = test_inference_modified(global_model, test_loader, MLP)
        print(f"训练后: ACC={acc_after:.4f}, AEOD={fm_after['EOD']:.4f}, ASPD={fm_after['SPD']:.4f}")
        first_weight_after = list(global_model.state_dict().values())[0]
        print(f"训练后权重样本: {first_weight_after[0, :5]}")
    
    # 最终评估
    loss, acc, fairness_metrics, _ = test_inference_modified(global_model, test_loader, MLP)
    print(f"  结果: ACC={acc:.4f}, AEOD={fairness_metrics['EOD']:.4f}, ASPD={fairness_metrics['SPD']:.4f}\n")
    return {'accuracy': acc, 'aeod': fairness_metrics['EOD'], 'aspd': fairness_metrics['SPD']}

# 测试2个实验
print("开始测试...\n")
try:
    # 测试1: FedAvg + Benign
    result1 = run_single_experiment('FedAvg_RW', 'benign', ALPHA_IID)
    print("[OK] 测试1通过: FedAvg_RW + Benign\n")

    # 测试2: FLTrust + FOE
    result2 = run_single_experiment('FLTrust_RW', 'foe', ALPHA_IID)
    print("[OK] 测试2通过: FLTrust_RW + FOE\n")

    print("="*80)
    print("所有测试通过！没有错误。")
    print("="*80)

except Exception as e:
    print(f"\n[ERROR] 测试失败: {e}")
    import traceback
    traceback.print_exc()
