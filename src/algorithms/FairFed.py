# FairFed.py - 更新版本，确保FairFed_RW正常运行

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import copy
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import random
import matplotlib as mpl
import torch.nn.functional as F

# 定义敏感属性常量
A_PRIVILEGED = 1  # Male
A_UNPRIVILEGED = 0  # Female

# 导入公平性计算函数（需要在运行时设置）
compute_fairness_metrics = None

# 注意：导入路径已更新为相对导入
# from HYPERPARAMETERS import DEVICE, HYPERPARAMETERS, SEED, algorithms, attack_forms, MALICIOUS_CLIENTS
# from data import ...
# from function import ...

class Client:
    def __init__(self, client_id, data, sensitive_features, batch_size, learning_rate, model_class, input_size, attack_form=None, use_reweighting=True):
        """初始化客户端类

        参数:
            use_reweighting: 是否使用Reweighting权重（默认True）
        """
        self.client_id = client_id
        self.X = data["X"]
        self.y = data["y"]
        self.sensitive_features = data["sensitive"]
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.attack_form = attack_form
        self.is_malicious = self.client_id in MALICIOUS_CLIENTS and self.attack_form != "no_attack"

        # 根据use_reweighting决定是否使用样本权重
        if use_reweighting:
            self.sample_weights = data.get("sample_weights", None)
        else:
            self.sample_weights = None  # 不使用Reweighting

        # 分割训练集和验证集（90%训练，10%验证）
        train_size = int(0.9 * len(self.X))
        val_size = len(self.X) - train_size

        if self.sample_weights is not None:
            dataset = TensorDataset(
                self.X, self.y,
                torch.tensor(self.sensitive_features, dtype=torch.long).to(HYPERPARAMETERS['DEVICE']),
                torch.tensor(self.sample_weights, dtype=torch.float32).to(HYPERPARAMETERS['DEVICE'])
            )
        else:
            dataset = TensorDataset(
                self.X, self.y,
                torch.tensor(self.sensitive_features, dtype=torch.long).to(HYPERPARAMETERS['DEVICE'])
            )

        from torch.utils.data import random_split
        self.train_data, self.val_data = random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(SEED)
        )

        self.train_loader = DataLoader(self.train_data, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_data, batch_size=batch_size, shuffle=False)

        self.model = model_class(input_size, HYPERPARAMETERS['OUTPUT_SIZE']).to(HYPERPARAMETERS['DEVICE'])
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.CrossEntropyLoss(reduction='none')

        # 为恶意客户端添加攻击功能
        if self.is_malicious:
            self.poison_data()

    def poison_data(self):
        """恶意客户端根据攻击形式进行数据中毒攻击"""
        education = self.X[:, 3]
        income = self.y
        sex = self.X[:, 9]

        if self.attack_form == "attack_fair_1":
            # 攻击fair_1: Attribute-flipping fairness attack
            mask_income0 = (income == 0)
            mask_income1 = (income == 1)
            self.X[mask_income0, 9] = A_PRIVILEGED
            self.X[mask_income1, 9] = A_UNPRIVILEGED
            self.sensitive_features[mask_income0.cpu().numpy()] = A_PRIVILEGED
            self.sensitive_features[mask_income1.cpu().numpy()] = A_UNPRIVILEGED
            print(f"客户端 {self.client_id} 执行攻击fair_1: Attribute-flipping fairness attack 完成。")

        elif self.attack_form == "attack_fair_2":
            # 攻击fair_2: Hybrid-flipping fairness attack
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

            print(f"客户端 {self.client_id} 执行攻击fair_2: Hybrid-flipping fairness attack 完成。")

        elif self.attack_form == "attack_acc_0.5":
            # 攻击acc_0.5: 将模型权重乘以-0.5 (在local_train中处理)
            print(f"客户端 {self.client_id} 将执行攻击acc_0.5: 模型权重*-0.5。")

        elif self.attack_form == "attack_acc_LIE":
            # 攻击acc_LIE: LIE攻击 (在local_train中处理)
            print(f"客户端 {self.client_id} 将执行攻击acc_LIE: LIE攻击。")

        elif self.attack_form == "attack_super_mixed":
            # 超级混合攻击：同时执行attack_fair_2的数据中毒和attack_acc_0.5的权重攻击
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
            print(f"客户端 {self.client_id} 未执行任何攻击。")

    def apply_weight_attack(self, attack_multiplier):
        """将所有模型参数乘以指定的攻击系数"""
        state_dict = self.model.state_dict()
        for key in state_dict:
            state_dict[key] = attack_multiplier * state_dict[key]
        self.model.load_state_dict(state_dict)

    def local_train_fairfed(self, global_model):
        """执行本地训练（适用于FairFed_RW）"""
        # 加载全局模型权重
        self.model.load_state_dict(global_model.state_dict())
        self.model.train()

        total_loss = 0.0
        total_batches = 0

        # 本地训练，应用RW权重
        for epoch in range(HYPERPARAMETERS['LOCAL_EPOCHS']):
            for batch in self.train_loader:
                if self.sample_weights is not None:
                    X_batch, y_batch, _, sample_weights_batch = batch
                else:
                    X_batch, y_batch, _ = batch

                X_batch, y_batch = X_batch.to(HYPERPARAMETERS['DEVICE']), y_batch.to(HYPERPARAMETERS['DEVICE'])

                self.optimizer.zero_grad()
                logits = self.model(X_batch)
                loss = self.criterion(logits, y_batch)

                if self.sample_weights is not None:
                    sample_weights_batch = sample_weights_batch.to(HYPERPARAMETERS['DEVICE'])
                    loss = loss * sample_weights_batch
                    loss = loss.sum() / sample_weights_batch.sum()
                else:
                    loss = loss.mean()

                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                total_batches += 1

        avg_loss = total_loss / total_batches if total_batches > 0 else 0.0

        # 恶意客户端行为处理
        if self.is_malicious:
            if self.attack_form == "attack_acc_0.5":
                self.apply_weight_attack(HYPERPARAMETERS['W_attack_0_5'])
            elif self.attack_form == "attack_super_mixed":
                self.apply_weight_attack(HYPERPARAMETERS['W_attack_0_5'])

        # 验证阶段 - 计算本地EOD
        spd, eod, local_acc = self.evaluate()

        return self.model.state_dict(), eod, local_acc, avg_loss

    def evaluate(self):
        """验证阶段: 计算本地EOD和SPD以及验证准确率"""
        self.model.eval()
        y_true, y_pred, sensitive_vals = [], [], []

        with torch.no_grad():
            for batch in self.val_loader:
                if self.sample_weights is not None:
                    X_batch, y_batch, sensitive_batch, _ = batch
                else:
                    X_batch, y_batch, sensitive_batch = batch
                X_batch, y_batch = X_batch.to(HYPERPARAMETERS['DEVICE']), y_batch.to(HYPERPARAMETERS['DEVICE'])
                logits = self.model(X_batch)
                preds = torch.argmax(logits, dim=1)
                y_pred.extend(preds.cpu().numpy())
                y_true.extend(y_batch.cpu().numpy())
                sensitive_vals.extend(sensitive_batch.cpu().numpy())

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        sensitive_vals = np.array(sensitive_vals)

        # 计算公平性指标
        fairness_metrics = compute_fairness_metrics(y_true, y_pred, sensitive_vals)
        spd = fairness_metrics["SPD"]
        eod = fairness_metrics["EOD"]

        # 计算本地验证准确率
        local_acc = accuracy_score(y_true, y_pred)
        return spd, eod, local_acc

class Server:
    def __init__(self, global_model, clients, algorithm, hyperparams):
        """初始化服务器类"""
        self.algorithm = algorithm
        self.hyperparams = hyperparams
        self.global_model = copy.deepcopy(global_model)
        self.clients = clients

        # 初始化FairFed权重
        self.bar_weights = {client.client_id: len(client.X) for client in self.clients}
        self.total_data = sum(self.bar_weights.values())
        self.bar_weights = {k: v / self.total_data for k, v in self.bar_weights.items()}
        self.global_weights = self.bar_weights.copy()
        self.global_eod = 0.0
        self.round_num = 0

    def aggregate(self, client_updates, deltas):
        """聚合客户端模型更新并更新全局模型权重"""
        if 'FairFed' in self.algorithm:
            aggregated_weights = self._fairfed_aggregate(client_updates, deltas)
        else:
            raise ValueError(f"未知的算法: {self.algorithm}")

        self.global_model.load_state_dict(aggregated_weights)

    def _fairfed_aggregate(self, client_updates, deltas):
        """FairFed算法的聚合方法，根据EOD调整客户端权重"""
        beta = self.hyperparams.get('BETA', 1)
        if len(deltas) == 0:
            avg_delta = 0
        else:
            avg_delta = np.mean(list(deltas.values()))

        # 计算每个客户端的新权重
        new_bar_weights = {}
        for client_id in self.bar_weights.keys():
            delta = deltas.get(client_id, 0)
            new_weight = self.bar_weights[client_id] - beta * (delta - avg_delta)
            new_weight = max(new_weight, 0)  # 确保权重不为负
            new_bar_weights[client_id] = new_weight

        # 更新bar_weights
        self.bar_weights = new_bar_weights

        # 归一化权重
        total_bar_weight = sum(self.bar_weights.values())
        if total_bar_weight == 0:
            weights = {client_id: 1 / len(self.bar_weights) for client_id in self.bar_weights}
        else:
            weights = {client_id: bar_weight / total_bar_weight for client_id, bar_weight in self.bar_weights.items()}

        # 基于权重进行加权聚合
        aggregated_weights = {}
        global_state_dict = self.global_model.state_dict()
        for name in global_state_dict.keys():
            aggregated_weights[name] = torch.zeros_like(global_state_dict[name]).to(DEVICE)
            for client_id, state_dict in client_updates.items():
                aggregated_weights[name] += weights[client_id] * state_dict[name]

        return aggregated_weights

    def run_round(self, round_num, test_df, y_test_values, model_class):
        """运行一轮全局训练"""
        client_updates = {}
        deltas = {}
        local_eods = []
        local_accs = []
        local_losses = []

        # 客户端本地训练
        for client in self.clients:
            local_weights, eod, local_acc, loss = client.local_train_fairfed(self.global_model)
            client_updates[client.client_id] = local_weights
            local_eods.append(eod)
            local_accs.append(local_acc)
            local_losses.append(loss)
            deltas[client.client_id] = abs(eod - self.global_eod)

        # 聚合模型更新
        self.aggregate(client_updates, deltas)

        # 更新全局EOD
        self.round_num += 1
        self.global_eod = np.mean(local_eods) if len(local_eods) > 0 else 0.0

        # 进行全局测试
        loss_test, accuracy, fairness_metrics, per_category = test_inference_modified(
            self.global_model,
            test_loader,
            model_class
        )

        return accuracy, fairness_metrics['EOD'], fairness_metrics['SPD'], per_category