import sys
sys.path.insert(0, 'D:\GitHub\GuardFed-main')

# 快速测试：只运行2个算法，2个攻击，1个分布，5轮训练
NUM_ROUNDS = 5

# 读取并修改完整脚本
with open('scripts/reproduce_table4_complete.py', 'r', encoding='utf-8') as f:
    code = f.read()

# 替换配置
code = code.replace('NUM_ROUNDS = 50', 'NUM_ROUNDS = 5')
code = code.replace(
    "algorithms = {\n        'FedAvg': train_fedavg,\n        'Median': train_median,\n        # 其他算法使用FedAvg作为占位符\n        'FairFed': train_fedavg,\n        'FLTrust': train_fedavg,\n        'FairGuard': train_fedavg,\n        'Hybrid': train_fedavg,\n        'GuardFed': train_fedavg\n    }",
    "algorithms = {'FedAvg': train_fedavg, 'Median': train_median}"
)
code = code.replace(
    "attacks = {\n        'Benign': 'benign',\n        'F Flip': 'f_flip',\n        'FOE': 'foe',\n        'S-DFA': 's_dfa',\n        'Sp-DFA': 'sp_dfa'\n    }",
    "attacks = {'Benign': 'benign', 'FOE': 'foe'}"
)
code = code.replace(
    "distributions = {\n        'IID': ALPHA_IID,\n        'non-IID': ALPHA_NON_IID\n    }",
    "distributions = {'IID': ALPHA_IID}"
)
code = code.replace('input("是否开始运行完整实验? (y/n): ")', '"y"')

exec(code)
