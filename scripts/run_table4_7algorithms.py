#!/usr/bin/env python3
# 运行表4所需的7个算法
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from datetime import datetime
from collections import defaultdict
import json

# 配置
NUM_ROUNDS = 70  # 改为70轮
NUM_CLIENTS = 20
NUM_MALICIOUS = 4
ALPHA_IID = 5000
ALPHA_NON_IID = 5

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

# 设置HYPERPARAMETERS到各个模块
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
print("运行表4所需的7个算法")
print("="*80)
print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"配置: {NUM_ROUNDS}轮, {NUM_CLIENTS}客户端, {NUM_MALICIOUS}恶意\n")

def run_single_experiment(algorithm, attack, alpha, dataset_name='adult'):
    """运行单个实验"""
    # 加载数据
    data_loader = DatasetLoader(dataset_name=dataset_name, seed=SEED, device=HYPERPARAMETERS['DEVICE'])
    data_info = data_loader.get_info()

    # 设置全局变量
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

    # 确定是否使用RW
    use_reweighting = '_RW' in algorithm

    # 创建客户端
    clients = []
    for i in range(NUM_CLIENTS):
        idx = client_data_indices[i]
        X_client, y_client = X_train[idx], y_train[idx]
        sex_client = sex_train[idx] if torch.is_tensor(sex_train) else sex_train[idx]

        sex_np = sex_client.cpu().numpy() if torch.is_tensor(sex_client) else sex_client
        y_np = y_client.cpu().numpy()
        weights = np.array([reweighing_weights.get((s, c), 1.0) for s, c in zip(sex_np, y_np)])

        # 攻击映射
        attack_map = {'benign': 'no_attack', 'f_flip': 'attack_fair_1', 'foe': 'attack_acc_0.5',
                     's_dfa': 'attack_super_mixed', 'sp_dfa': 'mixed'}
        attack_form = attack_map.get(attack, 'no_attack')

        client_data = {'X': X_client, 'y': y_client, 'sensitive': sex_np, 'sample_weights': weights}

        # 根据算法选择Client类
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
        else:  # FedAvg
            client = FedAvgClient(
                i, client_data, sex_np, HYPERPARAMETERS['BATCH_SIZE'],
                HYPERPARAMETERS['LEARNING_RATES']['FedAvg_RW'][0], MLP,
                data_info['num_features'], attack_form,
                use_reweighting=use_reweighting
            )
        clients.append(client)

    # 初始化全局模型
    global_model = MLP(num_features=data_info['num_features'], num_classes=2, seed=SEED).to(HYPERPARAMETERS['DEVICE'])

    # 准备服务器数据
    server_data = None
    if 'FLTrust' in algorithm:
        num_server = int(len(data_loader.train_df) * 0.1)
        server_data = data_loader.train_df.sample(n=num_server, random_state=SEED)

    # 创建服务器
    if 'FLTrust' in algorithm:
        server = FLTrustServer(global_model, clients, algorithm, HYPERPARAMETERS, server_data=server_data)
    elif 'FairFed' in algorithm:
        server = FairFedServer(global_model, clients, algorithm, HYPERPARAMETERS)
    elif 'Medium' in algorithm:
        server = MediumServer(global_model, clients, algorithm, HYPERPARAMETERS)
    else:  # FedAvg
        server = FedAvgServer(global_model, clients, algorithm, HYPERPARAMETERS)

    # 设置test_loader
    FedAvg_module.test_loader = test_loader
    FLTrust_module.test_loader = test_loader
    FairFed_module.test_loader = test_loader
    Medium_module.test_loader = test_loader

    # 训练
    for round_idx in range(NUM_ROUNDS):
        server.run_round(round_idx, None, None, MLP)
        global_model.load_state_dict(server.global_model.state_dict())

    # 最终评估
    loss, acc, fairness_metrics, _ = test_inference_modified(global_model, test_loader, MLP)
    return {'accuracy': acc, 'aeod': fairness_metrics['EOD'], 'aspd': fairness_metrics['SPD']}

def run_all_experiments():
    # 只运行表4需要的7个算法
    algorithms = [
        'FedAvg',              # 1. FedAvg (无RW)
        'FairFed',             # 2. FairFed (无RW)
        'Median',              # 3. Median (无RW)
        'FLTrust',             # 4. FLTrust (无RW)
        'FedAvg_FairG',        # 5. FairGuard (无RW)
        'FLTrust_FairG',       # 6. FLTrust + FairGuard (无RW)
        'FedAvg_RW_FairCosG',  # 7. GuardFed (已训练)
    ]

    attacks = {'Benign': 'benign', 'F Flip': 'f_flip', 'FOE': 'foe', 'S-DFA': 's_dfa', 'Sp-DFA': 'sp_dfa'}
    distributions = {'IID': ALPHA_IID, 'non-IID': ALPHA_NON_IID}

    all_results = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    total = len(algorithms) * len(attacks) * len(distributions)
    current = 0

    print(f"总实验数: {total}\n")

    for dist_name, alpha in distributions.items():
        print(f"\n{'='*80}\n数据分布: {dist_name} (α={alpha})\n{'='*80}\n")
        for algo in algorithms:
            for attack_name, attack_code in attacks.items():
                current += 1
                print(f"[{current}/{total}] {algo} | {attack_name} | {dist_name}")
                try:
                    start = datetime.now()
                    results = run_single_experiment(algo, attack_code, alpha)
                    elapsed = (datetime.now() - start).total_seconds()
                    all_results[dist_name][algo][attack_name] = results
                    print(f"    结果: ACC={results['accuracy']:.4f}, AEOD={results['aeod']:.4f}, ASPD={results['aspd']:.4f} ({elapsed:.1f}s)\n")
                except Exception as e:
                    print(f"    错误: {e}\n")
                    import traceback
                    traceback.print_exc()
                    all_results[dist_name][algo][attack_name] = {'accuracy': 0.0, 'aeod': 0.0, 'aspd': 0.0, 'error': str(e)}

    return all_results

if __name__ == "__main__":
    print("\n开始实验...\n")
    try:
        results = run_all_experiments()

        # 保存结果
        output_path = 'D:/GitHub/GuardFed-main/results/table4_7algorithms.json'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n结果已保存到: {output_path}")

        print("\n" + "="*80 + "\n所有实验完成！\n" + "="*80)
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n" + "="*80)
