# HYPERPARAMETERS.py - 统一命名规范版本
# 所有命名与论文保持一致，便于复现实验

import torch
import random
import numpy as np

# 设置随机种子和设备
SEED = 123
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

DEVICE = torch.device('cpu')  # 或者 'cuda' 如果有GPU

# 假设 sensitive 和 y 各有 2 个类别，则 num_groups = 2 * 2 = 4
HYPERPARAMETERS = {
    'ALPHA_DIRICHLET': [1],            # Dirichlet 分布的浓度参数，将在实验中动态修改
    'NUM_GLOBAL_ROUNDS': 50,               # 全局训练轮次
    'CLIENT_RATIO': 1.0,                   # 每轮选取的客户端比例 (100%)
    'BATCH_SIZE': 256,                     # 本地训练的批量大小
    'LEARNING_RATES': {                    # 各算法的学习率
        'FedAvg': [1e-2],
        'FedAvg_RW': [1e-2],
        'FedAvg_FairG': [1e-2],
        'FedAvg_RW_FairG': [1e-2],
        'FedAvg_RW_FairCosG': [1e-2],
        'FLTrust': [1e-2],
        'FLTrust_RW': [1e-2],
        'FLTrust_FairG': [1e-2],
        'FLTrust_RW_FairG': [1e-2],
        'FLTrust_RW_FairCosG': [1e-2],
        'FairCos_RW': [1e-2],
        'FairFed_RW': [1e-2],              # 新增FairFed_RW算法
        'Medium_RW': [1e-2],               # 新增Medium_RW算法
    },
    'BETA': 1,                             # FairFed 的公平性预算参数

    'NUM_CLIENTS': 20,                     # 客户端数量
    'OUTPUT_SIZE': 2,                      # 输出类别数（2 表示二分类）
    'DEVICE': DEVICE,                      # 设备配置
    'LOCAL_EPOCHS': 1,                     # 本地训练的 Epoch 数

    'FLTRUST_ALPHA': 0.5,                  # FLTrust 的更新权重因子
    'FLTRUST_BETA': 1e-2,                  # FLTrust 的优化器学习率

    'MIN_CLIENT_SIZE_RATIO': 0.01,         # 每个客户端最小数据量的比例
    'SERVER_EPOCHS': 1,                    # 服务器端训练的 Epoch 数
    'SERVER_DATA_RATIO': 0.1,              # 服务器数据比例，可以是 0.1, 0.05, 0.01, 0.005, 0.001

    'INPUT_SIZE': 20,                      # 输入特征维度，根据实际数据集调整
    'W_attack': -1,                        # 攻击权重 w = -1*w（假设为负值以实现反转）
    'W_attack_0_5': -0.5,                  # attack_acc_0.5的攻击权重
    'LIE_Z': 2,                           # LIE攻击的强度参数z (增大以确保攻击效果明显)
    'SEX_FLIP_PROPORTION': 1.0,            # 默认100%
    'LABEL_FLIP_PROPORTION': 1.0,          # 默认100%

    'num_classes': 2,                      # 输出类别数
    'num_groups': 4,                       # 修改为 4，假设 (sensitive, y) 有 4 组
    'FB_UPDATE_INTERVAL': 1,               # λ参数更新的间隔轮次
    'alpha': 0.1,                          # 用于更新 λ 的学习率因子
    'INITIAL_LAMBDA': 1.0,                 # 初始 lambda 值

    # 添加FairG相关超参数
    'FairG_epoch': 1,                      # 预热轮数，不进行FairG筛选的轮数
    'R': 500,                              # FairG生成的参考样本数量

    # 添加K-Means聚类相关参数
    'KMEANS_N_CLUSTERS': 2,                # 聚类的簇数量固定为2
    'KMEANS_INIT': 'random',            # 聚类的初始化方法
    'KMEANS_RANDOM_STATE': SEED,           # 聚类的随机种子
    'KMEANS_MAX_ITER': 100,                # 聚类的最大迭代次数
    'KMEANS_N_INIT': 10,                   # 聚类的初始次数
    'KMEANS_STANDARDIZE': False,            # 是否标准化可疑分数
    'KMEANS_TAU_RATIO': 0.005,             # τ的比例因子，tau = 0.05 * R

    # FairCosG相关超参数
    'FAIRCOSG_LAMBDA_VALUES': [20],        # 只测试λ=20
    'FAIRCOSG_EOD_TOLERANCE': 0.0,         # EOD容忍度：小于此值的EOD不被惩罚
    'FAIRCOSG_SCORE_THRESHOLD': 0.2,       # FairCosG分数阈值
    'FAIRCOSG_ALPHA': 0.2,                 # FairCosG的α参数
    'FAIRCOSG_BETA': 1e-2,                 # FairCosG的优化器学习率
    'FAIRCOSG_SERVER_EPOCHS': 1,           # FairCosG服务器端训练的Epoch数

    # 保留原FairCos相关超参数（用于向后兼容）
    'FAIRCOS_LAMBDA': 1.0,                  # 默认λ参数
    'FAIRCOS_LAMBDA_VALUES': [1],           # 只测试λ=1
    'FAIRCOS_EOD_TOLERANCE': 0.0,          # EOD容忍度：小于此值的EOD不被惩罚
    'FAIRCOS_SCORE_THRESHOLD': 0.0,        # FairCos分数阈值：设为0，所有客户端都参与聚合
    'FAIRCOS_ALPHA': 0.2,                  # FairCos的α参数
    'FAIRCOS_ALPHA_VALUES': [0.1, 0.2, 0.3, 0.4],  # 修改为用户指定的4个alpha值
}

# ============================================================================
# 命名规范：与论文保持一致
# ============================================================================

# 算法命名映射（论文名称 -> 代码实现）
ALGORITHM_MAPPING = {
    # 论文表格中的方法名称
    'FedAvg': 'FedAvg_RW',                    # FedAvg + Reweighting
    'FairFed': 'FairFed_RW',                  # FairFed + Reweighting
    'Median': 'Medium_RW',                    # Median + Reweighting
    'FLTrust': 'FLTrust_RW',                  # FLTrust + Reweighting
    'FairGuard': 'FedAvg_RW_FairG',           # FedAvg + RW + FairG
    'FLTrust+FairGuard': 'FLTrust_RW_FairG',  # FLTrust + RW + FairG
    'GuardFed': 'FedAvg_RW_FairCosG',         # GuardFed (我们的方法)
}

# 攻击类型命名映射（论文名称 -> 代码实现）
ATTACK_MAPPING = {
    # 论文表格中的攻击名称
    'Benign': 'no_attack',                    # 无攻击
    'F Flip': 'label_flip',                   # Label-flipping attack
    'FOE': 'foe',                             # FOE attack (权重×-0.5)
    'S-DFA': 's_dfa',                         # Synchronized Dual-Face Attack
    'Sp-DFA': 'sp_dfa',                       # Split Dual-Face Attack
}

# 代码内部使用的攻击实现名称
ATTACK_IMPLEMENTATIONS = {
    'no_attack': 'no_attack',                 # 无攻击
    'label_flip': 'attack_fair_1',            # 简单标签翻转
    'foe': 'attack_acc_0.5',                  # FOE攻击（权重×-0.5）
    's_dfa': 'attack_super_mixed',            # S-DFA（数据中毒+权重攻击）
    'sp_dfa': 'mixed',                        # Sp-DFA（分裂攻击）

    # 辅助攻击（用于构建混合攻击）
    'hybrid_flip': 'attack_fair_2',           # 混合翻转（用于Sp-DFA）
    'lie': 'attack_acc_LIE',                  # LIE攻击（备用）
}

# 默认恶意客户端配置（论文中使用4个恶意客户端，占20%）
NUM_MALICIOUS_CLIENTS = 4
MALICIOUS_CLIENTS = list(range(NUM_MALICIOUS_CLIENTS))  # [0, 1, 2, 3]

# 所有可用的算法列表（用于实验）
ALL_ALGORITHMS = [
    'FedAvg_RW',
    'FairFed_RW',
    'Medium_RW',
    'FLTrust_RW',
    'FedAvg_RW_FairG',
    'FLTrust_RW_FairG',
    'FedAvg_RW_FairCosG',
]

# 所有可用的攻击类型（用于实验）
ALL_ATTACKS = [
    'no_attack',
    'label_flip',
    'foe',
    's_dfa',
    'sp_dfa',
]

# 向后兼容：保留旧的命名
algorithms = ['FedAvg_RW_FairCosG']  # 默认测试算法
attack_forms = ["no_attack", "label_flip", "hybrid_flip", "foe", "lie", "s_dfa"]

# FLTrust 服务器数据比例设置
FLTRUST_SERVER_RATIOS = [0.1]

# ============================================================================
# 辅助函数：命名转换
# ============================================================================

def get_algorithm_code_name(paper_name):
    """
    将论文中的算法名称转换为代码实现名称

    Args:
        paper_name: 论文中的算法名称（如 'FedAvg', 'GuardFed'）

    Returns:
        代码实现名称（如 'FedAvg_RW', 'FedAvg_RW_FairCosG'）
    """
    return ALGORITHM_MAPPING.get(paper_name, paper_name)

def get_attack_code_name(paper_name):
    """
    将论文中的攻击名称转换为代码实现名称

    Args:
        paper_name: 论文中的攻击名称（如 'Benign', 'F Flip', 'S-DFA'）

    Returns:
        代码实现名称（如 'no_attack', 'label_flip', 's_dfa'）
    """
    return ATTACK_MAPPING.get(paper_name, paper_name)

def get_attack_implementation(attack_name):
    """
    获取攻击的具体实现名称（用于Client类）

    Args:
        attack_name: 统一的攻击名称（如 'no_attack', 'label_flip', 's_dfa'）

    Returns:
        实现名称（如 'no_attack', 'attack_fair_1', 'attack_super_mixed'）
    """
    return ATTACK_IMPLEMENTATIONS.get(attack_name, attack_name)

def get_paper_algorithm_name(code_name):
    """
    将代码实现名称转换回论文中的算法名称（用于输出结果）

    Args:
        code_name: 代码实现名称（如 'FedAvg_RW', 'FedAvg_RW_FairCosG'）

    Returns:
        论文名称（如 'FedAvg', 'GuardFed'）
    """
    reverse_mapping = {v: k for k, v in ALGORITHM_MAPPING.items()}
    return reverse_mapping.get(code_name, code_name)

def get_paper_attack_name(code_name):
    """
    将代码实现名称转换回论文中的攻击名称（用于输出结果）

    Args:
        code_name: 代码实现名称（如 'no_attack', 'label_flip', 's_dfa'）

    Returns:
        论文名称（如 'Benign', 'F Flip', 'S-DFA'）
    """
    reverse_mapping = {v: k for k, v in ATTACK_MAPPING.items()}
    return reverse_mapping.get(code_name, code_name)

# ============================================================================
# 实验配置：用于复现论文表格
# ============================================================================

# 论文表格中的所有方法（按表格顺序）
TABLE_METHODS = [
    'FedAvg',           # Fairness (debias) algorithms
    'FairFed',          # Fairness (debias) algorithms
    'Median',           # Robust FL for general adversarial attacks
    'FLTrust',          # Robust FL for general adversarial attacks
    'FairGuard',        # Robust FL for fairness attacks
    'FLTrust+FairGuard',# Robust FL (hybrid)
    'GuardFed',         # Ours
]

# 论文表格中的所有攻击（按表格顺序）
TABLE_ATTACKS = [
    'Benign',           # 无攻击
    'F Flip',           # 公平性攻击
    'FOE',              # 性能攻击
    'S-DFA',            # 同步双面攻击
    'Sp-DFA',           # 分裂双面攻击
]

# 论文表格中的数据分布
TABLE_DISTRIBUTIONS = {
    'IID': 100,         # α=100 接近IID
    'non-IID': 1,       # α=1 强non-IID
}

# 实验配置（与论文一致）
EXPERIMENT_CONFIG = {
    'num_clients': 20,
    'num_malicious': 4,              # 20%恶意客户端
    'num_rounds': 50,
    'server_data_ratio': 0.1,        # 10%服务器数据
    'batch_size': 256,
    'learning_rate': 0.01,
    'local_epochs': 1,
}

# ============================================================================
# 打印命名规范说明
# ============================================================================

def print_naming_convention():
    """打印命名规范说明"""
    print("\n" + "="*80)
    print("GuardFed 命名规范")
    print("="*80)

    print("\n【算法命名映射】")
    print(f"{'论文名称':<20s} -> {'代码实现名称':<30s}")
    print("-"*80)
    for paper_name, code_name in ALGORITHM_MAPPING.items():
        print(f"{paper_name:<20s} -> {code_name:<30s}")

    print("\n【攻击命名映射】")
    print(f"{'论文名称':<20s} -> {'统一名称':<20s} -> {'实现名称':<30s}")
    print("-"*80)
    for paper_name, unified_name in ATTACK_MAPPING.items():
        impl_name = ATTACK_IMPLEMENTATIONS.get(unified_name, unified_name)
        print(f"{paper_name:<20s} -> {unified_name:<20s} -> {impl_name:<30s}")

    print("\n【实验配置】")
    print("-"*80)
    for key, value in EXPERIMENT_CONFIG.items():
        print(f"{key:<25s}: {value}")

    print("\n" + "="*80)

# 如果直接运行此文件，打印命名规范
if __name__ == "__main__":
    print_naming_convention()