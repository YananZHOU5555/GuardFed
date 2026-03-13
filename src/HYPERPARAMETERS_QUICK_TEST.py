# HYPERPARAMETERS_QUICK_TEST.py - 快速测试配置
# 用于快速验证代码完整性

import torch
import random
import numpy as np

# 设置随机种子和设备
SEED = 123
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

DEVICE = torch.device('cpu')

# 快速测试配置：最小化参数以加快运行速度
HYPERPARAMETERS = {
    'ALPHA_DIRICHLET': [1],
    'NUM_GLOBAL_ROUNDS': 3,                # 减少到3轮（原50轮）
    'CLIENT_RATIO': 1.0,
    'BATCH_SIZE': 64,                      # 减小批量大小（原256）
    'LEARNING_RATES': {
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
        'FairFed_RW': [1e-2],
        'Medium_RW': [1e-2],
    },
    'BETA': 1,

    'NUM_CLIENTS': 6,                      # 减少到6个客户端（原20个）
    'OUTPUT_SIZE': 2,
    'DEVICE': DEVICE,
    'LOCAL_EPOCHS': 1,

    'FLTRUST_ALPHA': 0.5,
    'FLTRUST_BETA': 1e-2,

    'MIN_CLIENT_SIZE_RATIO': 0.01,
    'SERVER_EPOCHS': 1,
    'SERVER_DATA_RATIO': 0.1,

    'INPUT_SIZE': 20,
    'W_attack': -1,
    'W_attack_0_5': -0.5,
    'LIE_Z': 2,
    'SEX_FLIP_PROPORTION': 1.0,
    'LABEL_FLIP_PROPORTION': 1.0,

    'num_classes': 2,
    'num_groups': 4,
    'FB_UPDATE_INTERVAL': 1,
    'alpha': 0.1,
    'INITIAL_LAMBDA': 1.0,

    'FairG_epoch': 1,
    'R': 100,                              # 减少参考样本数（原500）

    'KMEANS_N_CLUSTERS': 2,
    'KMEANS_INIT': 'random',
    'KMEANS_RANDOM_STATE': SEED,
    'KMEANS_MAX_ITER': 100,
    'KMEANS_N_INIT': 10,
    'KMEANS_STANDARDIZE': False,
    'KMEANS_TAU_RATIO': 0.005,

    'FAIRCOSG_LAMBDA_VALUES': [20],
    'FAIRCOSG_EOD_TOLERANCE': 0.0,
    'FAIRCOSG_SCORE_THRESHOLD': 0.2,
    'FAIRCOSG_ALPHA': 0.2,
    'FAIRCOSG_BETA': 1e-2,
    'FAIRCOSG_SERVER_EPOCHS': 1,

    'FAIRCOS_LAMBDA': 1.0,
    'FAIRCOS_LAMBDA_VALUES': [1],
    'FAIRCOS_EOD_TOLERANCE': 0.0,
    'FAIRCOS_SCORE_THRESHOLD': 0.0,
    'FAIRCOS_ALPHA': 0.2,
    'FAIRCOS_ALPHA_VALUES': [0.1, 0.2, 0.3, 0.4],
}

# 算法命名映射
ALGORITHM_MAPPING = {
    'FedAvg': 'FedAvg_RW',
    'FairFed': 'FairFed_RW',
    'Median': 'Medium_RW',
    'FLTrust': 'FLTrust_RW',
    'FairGuard': 'FedAvg_RW_FairG',
    'FLTrust+FairGuard': 'FLTrust_RW_FairG',
    'GuardFed': 'FedAvg_RW_FairCosG',
}

# 攻击类型命名映射
ATTACK_MAPPING = {
    'Benign': 'no_attack',
    'F Flip': 'label_flip',
    'FOE': 'foe',
    'S-DFA': 's_dfa',
    'Sp-DFA': 'sp_dfa',
}

# 代码内部使用的攻击实现名称
ATTACK_IMPLEMENTATIONS = {
    'no_attack': 'no_attack',
    'label_flip': 'attack_fair_1',
    'foe': 'attack_acc_0.5',
    's_dfa': 'attack_super_mixed',
    'sp_dfa': 'mixed',
    'hybrid_flip': 'attack_fair_2',
    'lie': 'attack_acc_LIE',
}

# 快速测试：只测试2个恶意客户端（原4个）
NUM_MALICIOUS_CLIENTS = 2
MALICIOUS_CLIENTS = list(range(NUM_MALICIOUS_CLIENTS))

# 快速测试：只测试核心算法
ALL_ALGORITHMS = [
    'FedAvg_RW',
    'FedAvg_RW_FairCosG',  # GuardFed核心算法
]

# 快速测试：只测试2种攻击
ALL_ATTACKS = [
    'no_attack',
    'label_flip',
]

algorithms = ['FedAvg_RW_FairCosG']
attack_forms = ["no_attack", "label_flip"]

FLTRUST_SERVER_RATIOS = [0.1]

# 实验配置（快速测试版本）
EXPERIMENT_CONFIG = {
    'num_clients': 6,                      # 减少客户端数量
    'num_malicious': 2,                    # 减少恶意客户端
    'num_rounds': 3,                       # 减少训练轮次
    'server_data_ratio': 0.1,
    'batch_size': 64,                      # 减小批量大小
    'learning_rate': 0.01,
    'local_epochs': 1,
}

# 辅助函数
def get_algorithm_code_name(paper_name):
    return ALGORITHM_MAPPING.get(paper_name, paper_name)

def get_attack_code_name(paper_name):
    return ATTACK_MAPPING.get(paper_name, paper_name)

def get_attack_implementation(attack_name):
    return ATTACK_IMPLEMENTATIONS.get(attack_name, attack_name)

def get_paper_algorithm_name(code_name):
    reverse_mapping = {v: k for k, v in ALGORITHM_MAPPING.items()}
    return reverse_mapping.get(code_name, code_name)

def get_paper_attack_name(code_name):
    reverse_mapping = {v: k for k, v in ATTACK_MAPPING.items()}
    return reverse_mapping.get(code_name, code_name)

# 论文表格配置
TABLE_METHODS = [
    'FedAvg',
    'GuardFed',  # 快速测试只测试这两个
]

TABLE_ATTACKS = [
    'Benign',
    'F Flip',  # 快速测试只测试这两个
]

TABLE_DISTRIBUTIONS = {
    'IID': 100,
    'non-IID': 1,
}

def print_naming_convention():
    print("\n" + "="*80)
    print("GuardFed 快速测试配置")
    print("="*80)
    print("\n【快速测试参数】")
    print("-"*80)
    print(f"训练轮次: {HYPERPARAMETERS['NUM_GLOBAL_ROUNDS']} (原50轮)")
    print(f"客户端数量: {HYPERPARAMETERS['NUM_CLIENTS']} (原20个)")
    print(f"恶意客户端: {NUM_MALICIOUS_CLIENTS} (原4个)")
    print(f"批量大小: {HYPERPARAMETERS['BATCH_SIZE']} (原256)")
    print(f"参考样本数: {HYPERPARAMETERS['R']} (原500)")
    print(f"测试算法: {ALL_ALGORITHMS}")
    print(f"测试攻击: {ALL_ATTACKS}")
    print("\n" + "="*80)

if __name__ == "__main__":
    print_naming_convention()
