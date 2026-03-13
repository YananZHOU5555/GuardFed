#!/usr/bin/env python3
# scripts/run.py - 测试FedAvg_RW_FairCosG在不同恶意客户端数量下的效果

import sys
import os
# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import copy
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import math
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 导入项目模块
from src.HYPERPARAMETERS import HYPERPARAMETERS, SEED
from src.data_loader import DatasetLoader
from src.models.function import (
    compute_fairness_metrics,
    test_inference_modified,
    compute_reweighing_weights,
    assign_sample_weights_to_clients,
    MLP
)

# 导入所有算法
from src.algorithms.FedAvg import Server as FedAvgServer, Client as FedAvgClient
from src.algorithms.FLTrust import Server as FLTrustServer, Client as FLTrustClient
from src.algorithms.FairG import FairG
from src.algorithms.FairCosG import FairCosG

# ============================================================================
# 配置：数据集选择
# ============================================================================
DATASET_NAME = 'adult'  # 可选: 'adult' 或 'compas'

# 加载数据集
print(f"\n{'='*80}")
print(f"加载数据集: {DATASET_NAME.upper()}")
print(f"{'='*80}")

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
scaler = data_loader.scaler
categorical_columns = data_loader.categorical_columns
numerical_columns = data_loader.numerical_columns

# 敏感属性配置
SENSITIVE_COLUMN = data_loader.sensitive_column
A_PRIVILEGED = data_loader.A_PRIVILEGED
A_UNPRIVILEGED = data_loader.A_UNPRIVILEGED

# 创建测试数据加载器
test_loader = data_loader.create_test_loader(batch_size=HYPERPARAMETERS['BATCH_SIZE'])

# 设置输入特征维度
HYPERPARAMETERS['INPUT_SIZE'] = data_info['num_features']
print(f"设置输入特征维度: {HYPERPARAMETERS['INPUT_SIZE']}")
print(f"训练样本数: {data_info['num_train']}")
print(f"测试样本数: {data_info['num_test']}")
print(f"敏感属性: {SENSITIVE_COLUMN}")

# 定义要测试的算法（只有一个）
TEST_ALGORITHMS = ['FedAvg_RW_FairCosG']

# 定义要测试的攻击组合（只有一个）
TEST_ATTACK_COMBINATIONS = {
    "mixed": "mixed",  # 混合攻击：前一半公平攻击，后一半性能攻击
}

# 定义要测试的恶意客户端数量
MALICIOUS_CLIENT_NUMBERS = [2, 4, 6, 8, 10, 12, 14, 16, 18]

# 固定服务器数据比例为10%
FIXED_SERVER_DATA_RATIO = 0.1

# 创建支持新attack_super_mixed的客户端类
class CustomFedAvgClient(FedAvgClient):
    def poison_data(self):
        """恶意客户端根据攻击形式进行数据中毒攻击，修改super_mixed为attack_fair_2+权重攻击"""
        education = self.X[:, 3]
        income = self.y
        sex = self.X[:, 9]

        if self.attack_form == "attack_super_mixed":
            # 超级混合攻击：执行attack_fair_2的数据中毒 + attack_acc_0.5的权重攻击
            # 首先执行attack_fair_2的数据中毒
            mask_male_high_income = (sex == A_PRIVILEGED) & (income == 1)
            mask_female_low_income = (sex == A_UNPRIVILEGED) & (income == 0)

            num_male_high = mask_male_high_income.sum().item()
            num_female_low = mask_female_low_income.sum().item()

            half_male_high = num_male_high // 2
            half_female_low = num_female_low // 2

            indices_male_high = torch.nonzero(mask_male_high_income).squeeze()
            if half_male_high > 0 and len(indices_male_high) > 0:
                if indices_male_high.dim() == 0:
                    indices_male_high = indices_male_high.unsqueeze(0)
                selected_male_high = indices_male_high[torch.randperm(len(indices_male_high))[:half_male_high]]
                self.y[selected_male_high] = 0

            indices_female_low = torch.nonzero(mask_female_low_income).squeeze()
            if half_female_low > 0 and len(indices_female_low) > 0:
                if indices_female_low.dim() == 0:
                    indices_female_low = indices_female_low.unsqueeze(0)
                selected_female_low = indices_female_low[torch.randperm(len(indices_female_low))[:half_female_low]]
                self.X[selected_female_low, 9] = A_PRIVILEGED
                self.sensitive_features[selected_female_low.cpu().numpy()] = A_PRIVILEGED
                self.y[selected_female_low] = 0

            remaining_female_low = indices_female_low[half_female_low:] if len(indices_female_low) > half_female_low else torch.tensor([])
            if len(remaining_female_low) > 0:
                self.y[remaining_female_low] = 1

            remaining_male_high = indices_male_high[half_male_high:] if len(indices_male_high) > half_male_high else torch.tensor([])
            if len(remaining_male_high) > 0:
                self.X[remaining_male_high, 9] = A_UNPRIVILEGED
                self.sensitive_features[remaining_male_high.cpu().numpy()] = A_UNPRIVILEGED
                self.y[remaining_male_high] = 1

            print(f"客户端 {self.client_id} 执行攻击super_mixed: attack_fair_2数据中毒+权重攻击 完成。")
        else:
            # 调用父类的poison_data方法处理其他攻击
            super().poison_data()

    def local_train_fedavg(self, global_model):
        """执行本地训练，对于super_mixed攻击，在训练后应用权重攻击"""
        # 调用父类的训练方法
        model_weights, avg_loss = super().local_train_fedavg(global_model)
        
        # 如果是super_mixed攻击，额外应用权重攻击
        if self.is_malicious and self.attack_form == "attack_super_mixed":
            for key in model_weights:
                model_weights[key] = HYPERPARAMETERS['W_attack_0_5'] * model_weights[key]
            print(f"客户端 {self.client_id} 应用权重攻击（super_mixed）完成。")
        
        return model_weights, avg_loss

def create_malicious_clients_list(num_malicious):
    """根据恶意客户端数量创建恶意客户端列表"""
    return list(range(num_malicious))

def assign_attack_to_malicious_clients(attack_combination, malicious_clients_list):
    """为恶意客户端分配指定的攻击形式"""
    attack_assignment = {}
    num_malicious = len(malicious_clients_list)
    
    if attack_combination == "mixed":
        # 混合攻击：前一半恶意客户端执行公平攻击，后一半执行性能攻击
        for i, client_id in enumerate(malicious_clients_list):
            if i < num_malicious // 2:
                attack_assignment[client_id] = "attack_fair_2"
            else:
                attack_assignment[client_id] = "attack_acc_0.5"
        
        # 正常客户端不攻击
        for client_id in range(HYPERPARAMETERS['NUM_CLIENTS']):
            if client_id not in malicious_clients_list:
                attack_assignment[client_id] = "no_attack"
    
    return attack_assignment

def split_server_client_data(train_df, server_ratio):
    """分割服务器和客户端数据"""
    return data_loader.split_server_client_data(server_ratio)

def create_client_data_for_ratio(server_ratio):
    """为指定的服务器数据比例创建客户端数据字典"""
    print(f"\n创建服务器数据比例为 {server_ratio*100:.1f}% 的数据分割...")

    # 重新分割服务器和客户端数据
    server_df_new, client_df_new = split_server_client_data(train_df, server_ratio)
    
    # 重新标准化客户端数据
    client_df_new.reset_index(drop=True, inplace=True)
    X_train_client_new = client_df_new.drop('income', axis=1)
    y_train_client_new = client_df_new['income']
    
    # 标准化客户端数据（使用原scaler）
    X_train_client_new[numerical_columns] = scaler.transform(X_train_client_new[numerical_columns])
    
    # 转换为tensor
    X_train_client_tensor_new = torch.tensor(X_train_client_new.values, dtype=torch.float32).to(HYPERPARAMETERS['DEVICE'])
    y_train_client_tensor_new = torch.tensor(y_train_client_new.values, dtype=torch.long).to(HYPERPARAMETERS['DEVICE'])
    
    X_train_df_new = pd.DataFrame(X_train_client_tensor_new.cpu().numpy(), columns=client_df_new.drop('income', axis=1).columns, index=client_df_new.index)
    y_train_df_new = pd.Series(y_train_client_tensor_new.cpu().numpy(), index=client_df_new.index)
    
    # 使用相同的Dirichlet分布参数重新分配数据
    ALPHA = HYPERPARAMETERS['ALPHA_DIRICHLET'][0]
    NUM_CLIENTS = HYPERPARAMETERS['NUM_CLIENTS']
    
    # 分离特权群体和非特权群体
    privileged_indices = client_df_new[client_df_new[SENSITIVE_COLUMN] == A_PRIVILEGED].index
    unprivileged_indices = client_df_new[client_df_new[SENSITIVE_COLUMN] == A_UNPRIVILEGED].index
    
    privileged_X = X_train_df_new.loc[privileged_indices].reset_index(drop=True)
    privileged_y = y_train_df_new.loc[privileged_indices].reset_index(drop=True)
    
    unprivileged_X = X_train_df_new.loc[unprivileged_indices].reset_index(drop=True)
    unprivileged_y = y_train_df_new.loc[unprivileged_indices].reset_index(drop=True)
    
    # Dirichlet 分配比例（使用相同的随机种子确保一致性）
    np.random.seed(SEED)
    privileged_ratios = np.random.dirichlet([ALPHA] * NUM_CLIENTS, size=1)[0]
    unprivileged_ratios = np.random.dirichlet([ALPHA] * NUM_CLIENTS, size=1)[0]
    
    # 初始化客户端数据
    client_data_dict_new = {i: {"X": [], "y": [], "sensitive": []} for i in range(NUM_CLIENTS)}
    
    # 分配特权群体数据（男性）
    privileged_splits = (privileged_ratios * len(privileged_X)).astype(int)
    start_idx = 0
    for i, count in enumerate(privileged_splits):
        end_idx = start_idx + count
        if end_idx > len(privileged_X):
            end_idx = len(privileged_X)
        client_data_dict_new[i]["X"].append(privileged_X.iloc[start_idx:end_idx].values)
        client_data_dict_new[i]["y"].append(privileged_y.iloc[start_idx:end_idx].values)
        client_data_dict_new[i]["sensitive"].append(np.full(end_idx - start_idx, A_PRIVILEGED))
        start_idx = end_idx
    
    # 处理剩余的样本
    remaining = len(privileged_X) - start_idx
    if remaining > 0:
        client_data_dict_new[NUM_CLIENTS - 1]["X"].append(privileged_X.iloc[start_idx:].values)
        client_data_dict_new[NUM_CLIENTS - 1]["y"].append(privileged_y.iloc[start_idx:].values)
        client_data_dict_new[NUM_CLIENTS - 1]["sensitive"].append(np.full(remaining, A_PRIVILEGED))
    
    # 分配非特权群体数据（女性）
    unprivileged_splits = (unprivileged_ratios * len(unprivileged_X)).astype(int)
    start_idx = 0
    for i, count in enumerate(unprivileged_splits):
        end_idx = start_idx + count
        if end_idx > len(unprivileged_X):
            end_idx = len(unprivileged_X)
        client_data_dict_new[i]["X"].append(unprivileged_X.iloc[start_idx:end_idx].values)
        client_data_dict_new[i]["y"].append(unprivileged_y.iloc[start_idx:end_idx].values)
        client_data_dict_new[i]["sensitive"].append(np.full(end_idx - start_idx, A_UNPRIVILEGED))
        start_idx = end_idx
    
    # 处理剩余的样本
    remaining = len(unprivileged_X) - start_idx
    if remaining > 0:
        client_data_dict_new[NUM_CLIENTS - 1]["X"].append(unprivileged_X.iloc[start_idx:].values)
        client_data_dict_new[NUM_CLIENTS - 1]["y"].append(unprivileged_y.iloc[start_idx:].values)
        client_data_dict_new[NUM_CLIENTS - 1]["sensitive"].append(np.full(remaining, A_UNPRIVILEGED))
    
    # 合并数据并转换为Tensor
    for i in range(NUM_CLIENTS):
        if len(client_data_dict_new[i]["X"]) > 0:
            client_data_dict_new[i]["X"] = torch.tensor(np.vstack(client_data_dict_new[i]["X"]), dtype=torch.float32).to(HYPERPARAMETERS['DEVICE'])
        else:
            client_data_dict_new[i]["X"] = torch.empty(0, X_train_client_tensor_new.shape[1], dtype=torch.float32).to(HYPERPARAMETERS['DEVICE'])
        
        if len(client_data_dict_new[i]["y"]) > 0:
            client_data_dict_new[i]["y"] = torch.tensor(np.concatenate(client_data_dict_new[i]["y"]), dtype=torch.long).to(HYPERPARAMETERS['DEVICE'])
        else:
            client_data_dict_new[i]["y"] = torch.empty(0, dtype=torch.long).to(HYPERPARAMETERS['DEVICE'])
        
        if len(client_data_dict_new[i]["sensitive"]) > 0:
            client_data_dict_new[i]["sensitive"] = np.concatenate(client_data_dict_new[i]["sensitive"]).astype(int)
        else:
            client_data_dict_new[i]["sensitive"] = np.array([], dtype=int)
    
    return server_df_new, client_data_dict_new

def run_algorithm(algorithm, combination_name, attack_combination, num_malicious, server_df_current, client_data_dict_current, learning_rate=1e-2, lambda_val=None):
    """运行单个算法"""
    # 修改算法名称：包含恶意客户端数量信息
    algorithm_name = f'{algorithm}_{num_malicious}mal'
    
    print(f"\n===== 训练算法: {algorithm_name} | 攻击组合: {combination_name} | 恶意客户端数: {num_malicious} =====")
    
    # 创建当前实验的恶意客户端列表
    malicious_clients_list = create_malicious_clients_list(num_malicious)
    print(f"恶意客户端编号: {malicious_clients_list}")
    
    # 复制client_data_dict以避免修改原始数据
    clients_data = copy.deepcopy(client_data_dict_current)
    
    # 计算全局Reweighing权重
    reweighing_weights = compute_reweighing_weights(train_df, SENSITIVE_COLUMN, 'income')
    
    # 为算法分配样本权重
    assign_sample_weights_to_clients(clients_data, reweighing_weights, SENSITIVE_COLUMN, 'income')
    
    # 调试信息：验证重加权是否真的生效
    sample_client_id = 0  # 检查第一个客户端
    if 'sample_weights' in clients_data[sample_client_id]:
        weights = clients_data[sample_client_id]['sample_weights']
        unique_weights = np.unique(weights)
        print(f"    重加权验证 - 客户端{sample_client_id}权重范围: [{unique_weights.min():.4f}, {unique_weights.max():.4f}]")
    
    # 创建攻击分配
    attack_assignment = assign_attack_to_malicious_clients(attack_combination, malicious_clients_list)
    
    # 初始化客户端列表
    clients = []
    for client_id in range(HYPERPARAMETERS['NUM_CLIENTS']):
        client_info = clients_data[client_id]
        attack_form = attack_assignment.get(client_id, "no_attack")
        
        # 创建自定义客户端，传入当前实验的恶意客户端列表
        client = CustomFedAvgClient(
            client_id, client_info, client_info["sensitive"],
            HYPERPARAMETERS['BATCH_SIZE'], learning_rate, MLP,
            HYPERPARAMETERS['INPUT_SIZE'], attack_form=attack_form
        )
        
        # 重新设置is_malicious标志
        client.is_malicious = client_id in malicious_clients_list and attack_form != "no_attack"
        
        clients.append(client)
    
    # 初始化全局模型
    global_model = MLP(HYPERPARAMETERS['INPUT_SIZE'], HYPERPARAMETERS['OUTPUT_SIZE']).to(HYPERPARAMETERS['DEVICE'])
    
    # 创建FairCosG框架实例
    faircosg = FairCosG(server_data=server_df_current, lambda_param=lambda_val if lambda_val is not None else 20)
    
    # 创建服务器实例
    server = FedAvgServer(global_model, clients, algorithm, HYPERPARAMETERS, fairg=None, faircosg=faircosg, reweighing_weights=reweighing_weights)
    
    # 存储结果 - 修改为包含每轮详细数据
    detailed_results = []  # 用于CSV的详细数据
    results = {
        'rounds': [],
        'accuracy': [],
        'eod': [],
        'spd': []
    }
    
    # 运行所有轮次
    for round_num in range(HYPERPARAMETERS['NUM_GLOBAL_ROUNDS']):
        if round_num % 10 == 0:  # 每10轮打印一次
            print(f"--- {algorithm_name} | {combination_name} | 轮次 {round_num + 1} ---")
        
        accuracy, eod, spd, per_category, faircosg_scores = server.run_round(round_num, test_df, y_test.values, MLP)
        
        # 存储结果
        results['rounds'].append(round_num + 1)
        results['accuracy'].append(accuracy)
        results['eod'].append(abs(eod))  # 使用绝对值
        results['spd'].append(abs(spd))  # 使用绝对值
        
        # 存储详细数据用于CSV（包含恶意客户端数量信息）
        detailed_results.append({
            'Attack_Combination': combination_name,
            'Algorithm': algorithm,
            'Malicious_Clients': num_malicious,
            'Round': round_num + 1,
            'ACC': accuracy,
            'AEOD': abs(eod),
            'ASPD': abs(spd)
        })
        
        # 只在最后一轮打印详细信息
        if round_num == HYPERPARAMETERS['NUM_GLOBAL_ROUNDS'] - 1:
            print(f"{algorithm_name} | {combination_name} | 最终结果: ACC={accuracy:.6f}, AEOD={abs(eod):.6f}, ASPD={abs(spd):.6f}")
    
    return results, detailed_results

def run_all_experiments():
    """运行所有实验"""
    print(f"\n" + "="*80)
    print("开始FedAvg_RW_FairCosG不同恶意客户端数量实验（Mixed攻击）")
    print(f"攻击组合: {list(TEST_ATTACK_COMBINATIONS.keys())}")
    print(f"算法: {TEST_ALGORITHMS}")
    print(f"恶意客户端数量: {MALICIOUS_CLIENT_NUMBERS}")
    print(f"服务器数据比例: {FIXED_SERVER_DATA_RATIO*100:.1f}%")
    print("="*80)
    
    # 创建固定的服务器数据分割（10%）
    print(f"\n创建固定的服务器数据分割（{FIXED_SERVER_DATA_RATIO*100:.1f}%）...")
    server_df_fixed, client_data_dict_fixed = create_client_data_for_ratio(FIXED_SERVER_DATA_RATIO)
    
    # 存储所有详细结果用于CSV输出
    all_detailed_results = []
    # 存储汇总结果用于最终表格
    summary_results = []
    # 存储结果用于可视化
    all_results = {}
    
    for combination_name, attack_combination in TEST_ATTACK_COMBINATIONS.items():
        all_results[combination_name] = {}
        
        for num_malicious in MALICIOUS_CLIENT_NUMBERS:
            print(f"\n{'='*80}")
            print(f"测试恶意客户端数量: {num_malicious} / {HYPERPARAMETERS['NUM_CLIENTS']}")
            print(f"{'='*80}")
            
            for algorithm in TEST_ALGORITHMS:
                # 对于FairCosG算法，使用lambda=20
                for lambda_val in HYPERPARAMETERS['FAIRCOSG_LAMBDA_VALUES']:
                    print(f"\n{'='*60}")
                    print(f"运行 {algorithm} - {combination_name} - 恶意客户端: {num_malicious}")
                    print(f"{'='*60}")
                    
                    results, detailed_results = run_algorithm(
                        algorithm, combination_name, attack_combination, 
                        num_malicious, server_df_fixed, client_data_dict_fixed, 
                        lambda_val=lambda_val
                    )
                    
                    # 存储结果用于可视化（使用恶意客户端数量作为key的一部分）
                    result_key = f"{algorithm}_{num_malicious}mal"
                    all_results[combination_name][result_key] = results
                    
                    # 收集详细数据
                    all_detailed_results.extend(detailed_results)
                    
                    # 计算最后5轮的平均值用于汇总表格
                    last_5_rounds = {
                        'accuracy': np.mean(results['accuracy'][-5:]) if len(results['accuracy']) >= 5 else np.mean(results['accuracy']),
                        'eod': np.mean(results['eod'][-5:]) if len(results['eod']) >= 5 else np.mean(results['eod']),
                        'spd': np.mean(results['spd'][-5:]) if len(results['spd']) >= 5 else np.mean(results['spd'])
                    }
                    
                    summary_results.append({
                        'Attack_Combination': combination_name,
                        'Algorithm': algorithm,
                        'Malicious_Clients': num_malicious,
                        'Final_ACC': last_5_rounds['accuracy'],
                        'Final_AEOD': last_5_rounds['eod'],
                        'Final_ASPD': last_5_rounds['spd']
                    })
    
    return all_detailed_results, summary_results, all_results

def save_results_to_files(detailed_results, summary_results):
    """保存结果到CSV和Excel文件"""
    # 创建时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. 保存详细结果到CSV（用于画图）
    detailed_df = pd.DataFrame(detailed_results)
    csv_filename = f'faircosg_malicious_clients_detailed_{timestamp}.csv'
    detailed_df.to_csv(csv_filename, index=False, float_format='%.6f')
    print(f"\n详细结果已保存为CSV文件: {csv_filename}")
    
    # 2. 保存汇总结果到CSV
    summary_df = pd.DataFrame(summary_results)
    summary_csv_filename = f'faircosg_malicious_clients_summary_{timestamp}.csv'
    summary_df.to_csv(summary_csv_filename, index=False, float_format='%.6f')
    print(f"汇总结果已保存为CSV文件: {summary_csv_filename}")
    
    # 3. 创建Excel文件，包含两个Sheet
    excel_filename = f'faircosg_malicious_clients_results_{timestamp}.xlsx'
    try:
        with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
            # Sheet 1: 详细数据（每轮结果）
            detailed_df.to_excel(writer, sheet_name='Detailed_Results', index=False, float_format='%.6f')
            
            # Sheet 2: 汇总数据（最后5轮平均值）
            summary_df.to_excel(writer, sheet_name='Summary_Results', index=False, float_format='%.6f')
        
        print(f"完整实验结果已保存为Excel文件: {excel_filename}")
        print(f"  - Sheet1 'Detailed_Results': 包含所有轮次的详细数据")
        print(f"  - Sheet2 'Summary_Results': 包含最后5轮的平均值汇总")
    except ImportError:
        print(f"无法创建Excel文件（缺少openpyxl库），但CSV文件已成功保存")
        excel_filename = None
    
    # 4. 打印格式化的汇总表格（控制台显示）
    print(f"\n" + "="*120)
    print("实验结果汇总表格 (最后5轮平均值)")
    print("="*120)
    print(f"{'攻击组合':<15s} {'算法':<20s} {'恶意客户端':<12s} {'Final_ACC':<12s} {'Final_AEOD':<12s} {'Final_ASPD':<12s}")
    print("-" * 120)
    
    for combination in summary_df['Attack_Combination'].unique():
        comb_df = summary_df[summary_df['Attack_Combination'] == combination]
        print(f"\n{combination}:")
        print("-" * 120)
        for _, row in comb_df.iterrows():
            print(f"{'':15s} {row['Algorithm']:<20s} {row['Malicious_Clients']:<12d} "
                  f"{row['Final_ACC']:<12.6f} {row['Final_AEOD']:<12.6f} {row['Final_ASPD']:<12.6f}")
    
    return detailed_df, summary_df, csv_filename, summary_csv_filename, excel_filename

def plot_performance_comparison(all_results):
    """绘制性能对比图 - 按恶意客户端数量分组"""
    
    fig, axes = plt.subplots(3, 1, figsize=(15, 18))
    fig.suptitle('FedAvg_RW_FairCosG Performance under Different Numbers of Malicious Clients (Mixed Attack)', fontsize=20, fontweight='bold')
    
    # 定义不同恶意客户端数量的颜色
    colors = plt.cm.tab10(np.linspace(0, 1, len(MALICIOUS_CLIENT_NUMBERS)))
    
    combination_name = "mixed"  # 只有一个攻击组合
    
    # 第一行：Accuracy对比
    ax_acc = axes[0]
    for j, num_malicious in enumerate(MALICIOUS_CLIENT_NUMBERS):
        result_key = f"FedAvg_RW_FairCosG_{num_malicious}mal"
        if result_key in all_results[combination_name]:
            results = all_results[combination_name][result_key]
            ax_acc.plot(results['rounds'], results['accuracy'], 
                       color=colors[j], linewidth=2.5, 
                       label=f'{num_malicious} mal', alpha=0.9)
    
    ax_acc.set_title(f'{combination_name}: Accuracy', fontweight='bold', fontsize=14)
    ax_acc.set_xlabel('Round', fontsize=12)
    ax_acc.set_ylabel('Accuracy', fontsize=12)
    ax_acc.legend(title='Malicious Clients', fontsize=10, loc='best')
    ax_acc.grid(True, alpha=0.3)
    ax_acc.set_ylim([0.60, 0.90])
    
    # 第二行：EOD对比
    ax_eod = axes[1]
    for j, num_malicious in enumerate(MALICIOUS_CLIENT_NUMBERS):
        result_key = f"FedAvg_RW_FairCosG_{num_malicious}mal"
        if result_key in all_results[combination_name]:
            results = all_results[combination_name][result_key]
            ax_eod.plot(results['rounds'], results['eod'], 
                       color=colors[j], linewidth=2.5, 
                       label=f'{num_malicious} mal', alpha=0.9)
    
    ax_eod.set_title(f'{combination_name}: |EOD|', fontweight='bold', fontsize=14)
    ax_eod.set_xlabel('Round', fontsize=12)
    ax_eod.set_ylabel('|EOD|', fontsize=12)
    ax_eod.legend(title='Malicious Clients', fontsize=10, loc='best')
    ax_eod.grid(True, alpha=0.3)
    
    # 第三行：SPD对比
    ax_spd = axes[2]
    for j, num_malicious in enumerate(MALICIOUS_CLIENT_NUMBERS):
        result_key = f"FedAvg_RW_FairCosG_{num_malicious}mal"
        if result_key in all_results[combination_name]:
            results = all_results[combination_name][result_key]
            ax_spd.plot(results['rounds'], results['spd'], 
                       color=colors[j], linewidth=2.5, 
                       label=f'{num_malicious} mal', alpha=0.9)
    
    ax_spd.set_title(f'{combination_name}: |SPD|', fontweight='bold', fontsize=14)
    ax_spd.set_xlabel('Round', fontsize=12)
    ax_spd.set_ylabel('|SPD|', fontsize=12)
    ax_spd.legend(title='Malicious Clients', fontsize=10, loc='best')
    ax_spd.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

# 主程序
if __name__ == "__main__":
    print("="*80)
    print("FedAvg_RW_FairCosG不同恶意客户端数量实验（Mixed攻击）")
    print(f"算法：{TEST_ALGORITHMS}")
    print(f"攻击组合：{list(TEST_ATTACK_COMBINATIONS.keys())}")
    print(f"恶意客户端数量：{MALICIOUS_CLIENT_NUMBERS}")
    print(f"服务器数据比例：{FIXED_SERVER_DATA_RATIO*100:.1f}%")
    print(f"FairCosG Lambda值：{HYPERPARAMETERS['FAIRCOSG_LAMBDA_VALUES']}")
    print(f"训练轮次：{HYPERPARAMETERS['NUM_GLOBAL_ROUNDS']}")
    print("="*80)
    
    # 运行所有实验
    detailed_results, summary_results, all_results = run_all_experiments()
    
    # 保存结果到文件并显示表格
    detailed_df, summary_df, csv_filename, summary_csv_filename, excel_filename = save_results_to_files(detailed_results, summary_results)
    
    # 生成可视化图表
    print(f"\n{'='*80}")
    print("生成可视化图表...")
    print(f"{'='*80}")
    
    fig = plot_performance_comparison(all_results)
    
    # 保存图片
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f'faircosg_malicious_clients_comparison_{timestamp}.png'
    fig.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"性能对比图已保存为: {plot_filename}")
    
    # 显示图表
    plt.show()
    
    print("\n" + "="*80)
    print("所有实验完成！")
    print("="*80)
    print(f"\n实验总结:")
    print(f"- 测试了FedAvg_RW_FairCosG算法在{len(MALICIOUS_CLIENT_NUMBERS)}种不同恶意客户端数量下的表现")
    print(f"- 测试了Mixed攻击组合（前一半公平攻击，后一半性能攻击）")
    print(f"- 恶意客户端数量：{MALICIOUS_CLIENT_NUMBERS}")
    print(f"- 固定服务器数据比例：{FIXED_SERVER_DATA_RATIO*100:.1f}%")
    print(f"- 训练轮次：{HYPERPARAMETERS['NUM_GLOBAL_ROUNDS']}")
    print(f"- 记录指标：ACC（准确率）、AEOD（绝对EOD）、ASPD（绝对SPD）")
    print(f"\n生成的文件:")
    print(f"- 详细结果CSV文件：{csv_filename}")
    print(f"  * 包含每个恶意客户端数量、每个攻击组合、每轮的详细数据")
    print(f"  * 适合用于绘制训练过程曲线图")
    print(f"- 汇总结果CSV文件：{summary_csv_filename}")
    print(f"  * 包含最后5轮平均值的汇总数据")
    print(f"  * 适合用于不同恶意客户端数量的性能对比")
    if excel_filename:
        print(f"- Excel文件：{excel_filename}")
        print(f"  * 包含详细数据和汇总数据两个Sheet")
        print(f"  * 便于查看和进一步分析")
    print(f"- 可视化图表：{plot_filename}")
    print(f"  * 3行1列的性能对比图")
    print(f"  * 显示不同恶意客户端数量下Mixed攻击的表现")
    
    print(f"\n攻击组合说明:")
    print(f"- mixed: 前一半恶意客户端执行attack_fair_2（公平攻击），后一半执行attack_acc_0.5（性能攻击）")
    print(f"  例如：2个恶意客户端时，1个公平攻击+1个性能攻击")
    print(f"       18个恶意客户端时，9个公平攻击+9个性能攻击")
    
    print(f"\n实验目的:")
    print(f"- 研究FairCosG框架在不同恶意客户端数量下的防御效果")
    print(f"- 分析恶意客户端数量对FairCosG筛选能力的影响")
    print(f"- 评估FairCosG的防御能力上限")
    
    print(f"\n期望结果:")
    print(f"- 随着恶意客户端数量增加（2→18），攻击效果应该逐渐增强")
    print(f"- FairCosG应该能够识别并过滤大部分恶意客户端，但防御效果可能随恶意客户端增加而下降")
    print(f"- 混合攻击同时影响模型的准确性和公平性，观察FairCosG的综合防御能力")
    print(f"- 找出FairCosG能有效防御的恶意客户端数量上限")
    print(f"\n测试完成！")