#!/usr/bin/env python3
# 快速测试7个算法（1轮）
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from datetime import datetime

# 配置 - 只用1轮快速测试
NUM_ROUNDS = 1
NUM_CLIENTS = 20
NUM_MALICIOUS = 4
ALPHA_IID = 5000

# 设置全局变量
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

from src.HYPERPARAMETERS import HYPERPARAMETERS, SEED

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

from src.models.function import test_inference_modified as test_func
from src.models.function import compute_fairness_metrics
FedAvg_module.test_inference_modified = test_func
FLTrust_module.test_inference_modified = test_func
FairFed_module.test_inference_modified = test_func
Medium_module.test_inference_modified = test_func
FairFed_module.compute_fairness_metrics = compute_fairness_metrics

from src.data_loader import DatasetLoader
from src.models.function import compute_reweighing_weights, MLP, test_inference_modified
from src.algorithms.FedAvg import Server as FedAvgServer, Client as FedAvgClient
from src.algorithms.FLTrust import Server as FLTrustServer, Client as FLTrustClient
from src.algorithms.FairFed import Server as FairFedServer, Client as FairFedClient
from src.algorithms.Medium import Server as MediumServer, Client as MediumClient

print("\n" + "="*80)
print("快速测试7个算法（1轮）")
print("="*80)
print(f"配置: {NUM_ROUNDS}轮, {NUM_CLIENTS}客户端, {NUM_MALICIOUS}恶意\n")

def run_single_experiment(algorithm, attack, alpha):
    """运行单个实验"""
    print(f"测试: {algorithm} | {attack}")

    data_loader = DatasetLoader(dataset_name='adult', seed=SEED, device=HYPERPARAMETERS['DEVICE'])
    data_info = data_loader.get_info()

    FLTrust_module.scaler = data_loader.scaler
    FLTrust_module.numerical_columns = data_loader.numerical_columns

    tensors = data_loader.get_tensors()
    X_train, y_train, sex_train = tensors['X_train'], tensors['y_train'], tensors['sex_train']
    test_loader = data_loader.create_test_loader(batch_size=HYPERPARAMETERS['BATCH_SIZE'])

    num_train, num_classes = len(X_train), 2
    client_data_indices = [[] for _ in range(NUM_CLIENTS)]
    for k in range(num_classes):
        idx_k = np.where(y_train.cpu().numpy() == k)[0]
        np.random.shuffle(idx_k)
        proportions = np.random.dirichlet(np.repeat(alpha, NUM_CLIENTS))
        proportions = proportions / proportions.sum()
        proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
        client_data_indices = [idx_j + idx.tolist() for idx_j, idx in zip(client_data_indices, np.split(idx_k, proportions))]

    reweighing_weights = compute_reweighing_weights(data_loader.train_df, data_loader.sensitive_column, 'income')

    use_reweighting = '_RW' in algorithm

    clients = []
    for i in range(NUM_CLIENTS):
        idx = client_data_indices[i]
        X_client, y_client = X_train[idx], y_train[idx]
        sex_client = sex_train[idx] if torch.is_tensor(sex_train) else sex_train[idx]

        sex_np = sex_client.cpu().numpy() if torch.is_tensor(sex_client) else sex_client
        y_np = y_client.cpu().numpy()
        weights = np.array([reweighing_weights.get((s, c), 1.0) for s, c in zip(sex_np, y_np)])

        attack_map = {'benign': 'no_attack', 'f_flip': 'attack_fair_1', 'foe': 'attack_acc_0.5',
                     's_dfa': 'attack_super_mixed', 'sp_dfa': 'mixed'}
        attack_form = attack_map.get(attack, 'no_attack')

        client_data = {'X': X_client, 'y': y_client, 'sensitive': sex_np, 'sample_weights': weights}

        if 'FLTrust' in algorithm:
            client = FLTrustClient(
                i, client_data, sex_np, HYPERPARAMETERS['BATCH_SIZE'],
                HYPERPARAMETERS['LEARNING_RATES']['FLTrust_RW'][0], MLP,
                data_info['num_features'], attack_form,
                use_reweighting=use_reweighting
            )
        elif 'FairFed' in algorithm:
            client = FairFedClient(
                i, client_data, sex_np, HYPERPARAMETERS['BATCH_SIZE'],
                HYPERPARAMETERS['LEARNING_RATES']['FairFed_RW'][0], MLP,
                data_info['num_features'], attack_form,
                use_reweighting=use_reweighting
            )
        elif 'Medium' in algorithm:
            client = MediumClient(
                i, client_data, sex_np, HYPERPARAMETERS['BATCH_SIZE'],
                HYPERPARAMETERS['LEARNING_RATES']['FedAvg_RW'][0], MLP,
                data_info['num_features'], attack_form,
                use_reweighting=use_reweighting
            )
        else:
            client = FedAvgClient(
                i, client_data, sex_np, HYPERPARAMETERS['BATCH_SIZE'],
                HYPERPARAMETERS['LEARNING_RATES']['FedAvg_RW'][0], MLP,
                data_info['num_features'], attack_form,
                use_reweighting=use_reweighting
            )
        clients.append(client)

    global_model = MLP(num_features=data_info['num_features'], num_classes=2, seed=SEED).to(HYPERPARAMETERS['DEVICE'])

    server_data = None
    if 'FLTrust' in algorithm:
        num_server = int(len(data_loader.train_df) * 0.1)
        server_data = data_loader.train_df.sample(n=num_server, random_state=SEED)

    if 'FLTrust' in algorithm:
        server = FLTrustServer(global_model, clients, algorithm, HYPERPARAMETERS, server_data=server_data)
    elif 'FairFed' in algorithm:
        server = FairFedServer(global_model, clients, algorithm, HYPERPARAMETERS)
    elif 'Medium' in algorithm:
        server = MediumServer(global_model, clients, algorithm, HYPERPARAMETERS)
    else:
        server = FedAvgServer(global_model, clients, algorithm, HYPERPARAMETERS)

    FedAvg_module.test_loader = test_loader
    FLTrust_module.test_loader = test_loader
    FairFed_module.test_loader = test_loader
    Medium_module.test_loader = test_loader

    for round_idx in range(NUM_ROUNDS):
        server.run_round(round_idx, None, None, MLP)
        global_model.load_state_dict(server.global_model.state_dict())

    loss, acc, fairness_metrics, _ = test_inference_modified(global_model, test_loader, MLP)
    print(f"  结果: ACC={acc:.4f}, AEOD={fairness_metrics['EOD']:.4f}, ASPD={fairness_metrics['SPD']:.4f}\n")
    return {'accuracy': acc, 'aeod': fairness_metrics['EOD'], 'aspd': fairness_metrics['SPD']}

# 测试7个算法
algorithms = [
    'FedAvg',
    'FairFed',
    'Median',
    'FLTrust',
    'FedAvg_FairG',
    'FLTrust_FairG',
    'FedAvg_RW_FairCosG',
]

print("开始测试...\n")
results = {}
errors = {}

for algo in algorithms:
    try:
        result = run_single_experiment(algo, 'benign', ALPHA_IID)
        results[algo] = result
        print(f"[OK] {algo} 测试通过\n")
    except Exception as e:
        errors[algo] = str(e)
        print(f"[ERROR] {algo} 失败: {e}\n")
        import traceback
        traceback.print_exc()

print("="*80)
print(f"测试完成: {len(results)}/7 成功, {len(errors)}/7 失败")
if errors:
    print("\n失败的算法:")
    for algo, error in errors.items():
        print(f"  - {algo}: {error}")
print("="*80)
