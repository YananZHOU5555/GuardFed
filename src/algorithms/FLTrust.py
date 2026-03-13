# FLTrust.py

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
# 导入绘图所需库
import matplotlib.pyplot as plt
import seaborn as sns

# 定义敏感属性常量
A_PRIVILEGED = 1  # Male
A_UNPRIVILEGED = 0  # Female

# 注意：导入路径已更新为相对导入
# from FairG import FairG
# from FairCosG import FairCosG
# from HYPERPARAMETERS import DEVICE,HYPERPARAMETERS,SEED,algorithms,attack_forms,MALICIOUS_CLIENTS
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
        self.attack_form = attack_form  # 攻击形式
        self.is_malicious = self.client_id in MALICIOUS_CLIENTS and self.attack_form != "no_attack"

        # 根据use_reweighting决定是否使用样本权重
        if use_reweighting:
            self.sample_weights = data.get("sample_weights", None)
        else:
            self.sample_weights = None  # 不使用Reweighting

        # 分割训练集和验证集（90%训练，10%验证）
        train_size = int(0.9 * len(self.X))
        val_size = len(self.X) - train_size

        # 分割数据集，保持敏感特征
        if self.sample_weights is not None:
            dataset = TensorDataset(
                self.X,
                self.y,
                torch.tensor(self.sensitive_features, dtype=torch.long).to(HYPERPARAMETERS['DEVICE']),
                torch.tensor(self.sample_weights, dtype=torch.float32).to(HYPERPARAMETERS['DEVICE'])
            )
        else:
            dataset = TensorDataset(
                self.X,
                self.y,
                torch.tensor(self.sensitive_features, dtype=torch.long).to(HYPERPARAMETERS['DEVICE'])
            )

        self.train_data, self.val_data = random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(SEED)
        )

        # 数据加载器
        self.train_loader = DataLoader(self.train_data, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_data, batch_size=batch_size, shuffle=False)

        # 模型和优化器
        self.model = model_class(input_size, HYPERPARAMETERS['OUTPUT_SIZE']).to(HYPERPARAMETERS['DEVICE'])
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.CrossEntropyLoss(reduction='mean')  # 使用交叉熵损失

        # 为恶意客户端添加攻击功能
        if self.is_malicious:
            self.poison_data()

    def poison_data(self):
        """恶意客户端根据攻击形式进行数据中毒攻击"""
        education = self.X[:, 3]
        income = self.y
        sex = self.X[:, 9]

        if self.attack_form == "attack1":
            # 攻击1: Label-flipping fairness attack
            mask_male = (sex == A_PRIVILEGED)
            mask_female = (sex == A_UNPRIVILEGED)
            self.y[mask_male] = 0
            self.y[mask_female] = 1
            print(f"客户端 {self.client_id} 执行攻击1: Label-flipping fairness attack 完成。")

        elif self.attack_form == "attack2":
            # 攻击2: Attribute-flipping fairness attack
            mask_income0 = (income == 0)
            mask_income1 = (income == 1)
            self.X[mask_income0, 9] = A_PRIVILEGED
            self.X[mask_income1, 9] = A_UNPRIVILEGED
            self.sensitive_features[mask_income0.cpu().numpy()] = A_PRIVILEGED
            self.sensitive_features[mask_income1.cpu().numpy()] = A_UNPRIVILEGED
            print(f"客户端 {self.client_id} 执行攻击2: Attribute-flipping fairness attack 完成。")

        elif self.attack_form == "attack3":
            # 攻击3: Hybrid-flipping fairness attack
            mask_male_high_income = (sex == A_PRIVILEGED) & (income == 1)
            mask_female_low_income = (sex == A_UNPRIVILEGED) & (income == 0)

            num_male_high = mask_male_high_income.sum().item()
            num_female_low = mask_female_low_income.sum().item()

            half_male_high = num_male_high // 2
            half_female_low = num_female_low // 2

            indices_male_high = torch.nonzero(mask_male_high_income).squeeze()
            if half_male_high > 0:
                selected_male_high = indices_male_high[torch.randperm(len(indices_male_high))[:half_male_high]]
                self.y[selected_male_high] = 0

            indices_female_low = torch.nonzero(mask_female_low_income).squeeze()
            if half_female_low > 0:
                selected_female_low = indices_female_low[torch.randperm(len(indices_female_low))[:half_female_low]]
                self.X[selected_female_low, 9] = A_PRIVILEGED
                self.sensitive_features[selected_female_low.cpu().numpy()] = A_PRIVILEGED
                self.y[selected_female_low] = 0

            remaining_female_low = indices_female_low[half_female_low:]
            if len(remaining_female_low) > 0:
                self.y[remaining_female_low] = 1

            remaining_male_high = indices_male_high[half_male_high:]
            if len(remaining_male_high) > 0:
                self.X[remaining_male_high, 9] = A_UNPRIVILEGED
                self.sensitive_features[remaining_male_high.cpu().numpy()] = A_UNPRIVILEGED
                self.y[remaining_male_high] = 1

            print(f"客户端 {self.client_id} 执行攻击3: Hybrid-flipping fairness attack 完成。")

        elif self.attack_form == "attack4":
            # 攻击4: Double-flipping fairness attack
            mask_male_high_income = (sex == A_PRIVILEGED) & (income == 1)
            mask_female_low_income = (sex == A_UNPRIVILEGED) & (income == 0)

            self.X[mask_male_high_income, 9] = A_UNPRIVILEGED
            self.sensitive_features[mask_male_high_income.cpu().numpy()] = A_UNPRIVILEGED
            self.y[mask_male_high_income] = 0

            self.X[mask_female_low_income, 9] = A_PRIVILEGED
            self.sensitive_features[mask_female_low_income.cpu().numpy()] = A_PRIVILEGED
            self.y[mask_female_low_income] = 1

            print(f"客户端 {self.client_id} 执行攻击4: Double-flipping fairness attack 完成。")

        elif self.attack_form == "attack_fair_1":
            # 新攻击fair_1: Attribute-flipping fairness attack
            mask_income0 = (income == 0)
            mask_income1 = (income == 1)
            self.X[mask_income0, 9] = A_PRIVILEGED
            self.X[mask_income1, 9] = A_UNPRIVILEGED
            self.sensitive_features[mask_income0.cpu().numpy()] = A_PRIVILEGED
            self.sensitive_features[mask_income1.cpu().numpy()] = A_UNPRIVILEGED
            print(f"客户端 {self.client_id} 执行攻击fair_1: Attribute-flipping fairness attack 完成。")

        elif self.attack_form == "attack_fair_2":
            # 新攻击fair_2: Hybrid-flipping fairness attack
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
            # 新攻击acc_0.5: 将模型权重乘以-0.5 (在local_train中处理)
            print(f"客户端 {self.client_id} 将执行攻击acc_0.5: 模型权重*-0.5。")

        elif self.attack_form == "attack_acc_LIE":
            # 新攻击acc_LIE: LIE攻击 (在local_train中处理)
            print(f"客户端 {self.client_id} 将执行攻击acc_LIE: LIE攻击。")

        elif self.attack_form == "attack_super_mixed":
            # 超级混合攻击：同时执行attack_fair_2的数据中毒和attack_acc_0.5的权重攻击
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
            print(f"客户端 {self.client_id} 未执行任何攻击。")
            return

    def invert_model_weights(self):
        """
        将所有模型参数乘以HYPERPARAMETERS['W_attack']，适用于恶意客户端。
        """
        state_dict = self.model.state_dict()
        for key in state_dict:
            state_dict[key] = HYPERPARAMETERS['W_attack'] * state_dict[key]
        self.model.load_state_dict(state_dict)

    def apply_weight_attack(self, attack_multiplier):
        """
        将所有模型参数乘以指定的攻击系数
        """
        state_dict = self.model.state_dict()
        for key in state_dict:
            state_dict[key] = attack_multiplier * state_dict[key]
        self.model.load_state_dict(state_dict)

    def local_train_fltrust(self, global_model, global_model_state_dict, fltrust_alpha, fltrust_beta):
        """
        执行本地训练（适用于FLTrust和FLTrust_RW）。
        """
        # 加载全局模型权重
        self.model.load_state_dict(global_model.state_dict())
        self.model.train()  # 设置为训练模式

        # 使用 Adam 优化器，并应用 fltrust_beta 作为学习率
        optimizer = optim.Adam(self.model.parameters(), lr=fltrust_beta)
        criterion = nn.CrossEntropyLoss(reduction='mean')  # 'mean'

        total_loss = 0.0
        total_batches = 0

        # 本地训练
        for epoch in range(HYPERPARAMETERS['LOCAL_EPOCHS']):
            for batch in self.train_loader:
                if self.sample_weights is not None:
                    X_batch, y_batch, _, sample_weights_batch = batch
                    X_batch, y_batch, sample_weights_batch = X_batch.to(HYPERPARAMETERS['DEVICE']), y_batch.to(HYPERPARAMETERS['DEVICE']), sample_weights_batch.to(HYPERPARAMETERS['DEVICE'])
                    loss = criterion(self.model(X_batch), y_batch)
                    loss = (loss * sample_weights_batch).mean()
                else:
                    X_batch, y_batch, _ = batch
                    X_batch, y_batch = X_batch.to(HYPERPARAMETERS['DEVICE']), y_batch.to(HYPERPARAMETERS['DEVICE'])
                    loss = criterion(self.model(X_batch), y_batch)

                optimizer.zero_grad()  # 清除梯度
                loss.backward()  # 反向传播
                optimizer.step()  # 更新权重
                total_loss += loss.item()
                total_batches += 1

        avg_loss = total_loss / total_batches if total_batches > 0 else 0.0

        # 获取新的模型参数
        new_weights = self.model.state_dict()

        # 恶意客户端行为处理
        if self.is_malicious:
            if self.attack_form == "attack_acc":
                for key in new_weights:
                    new_weights[key] = HYPERPARAMETERS['W_attack'] * new_weights[key]
            elif self.attack_form == "attack_acc_0.5":
                for key in new_weights:
                    new_weights[key] = HYPERPARAMETERS['W_attack_0_5'] * new_weights[key]
            elif self.attack_form == "attack_super_mixed":
                # 对于super_mixed攻击，额外应用权重攻击
                for key in new_weights:
                    new_weights[key] = HYPERPARAMETERS['W_attack_0_5'] * new_weights[key]

        # 计算权重更新（delta）：new_weights - global_model_state_dict
        delta_weights = {k: new_weights[k] - global_model_state_dict[k] for k in global_model_state_dict.keys()}

        return delta_weights, avg_loss

class Server:
    def __init__(self, global_model, clients, algorithm, hyperparams, server_data=None, fairg=None, faircosg=None):
        """
        初始化服务器类。
        """
        self.algorithm = algorithm
        self.hyperparams = hyperparams
        self.global_model = copy.deepcopy(global_model)  # 全局模型
        self.clients = clients  # 客户端列表

        # 初始化FairG
        self.fairg = fairg
        if self.fairg is not None:
            self.current_round = 0

        # 初始化FairCosG
        self.faircosg = faircosg
        if self.faircosg is not None:
            self.current_round = 0

        # 初始化 FLTrust 特有的参数
        if self.algorithm in ['FLTrust', 'FLTrust_RW', 'FLTrust_FairG', 'FLTrust_RW_FairG', 'FLTrust_RW_FairCosG']:
            self.server_data = server_data  # 服务器数据集
            if self.server_data is not None:
                self.server_model = copy.deepcopy(global_model).to(HYPERPARAMETERS['DEVICE'])
                self.server_optimizer = optim.Adam(self.server_model.parameters(), lr=hyperparams['FLTRUST_BETA'])
                self.server_criterion = nn.CrossEntropyLoss()

    def aggregate(self, deltas=None, TS_ratio=None, client_discrimination_scores=None, faircosg_scores=None, excluded_clients_info=None):
        """
        聚合客户端的模型更新，更新全局模型权重。
        """
        if self.algorithm in ['FLTrust', 'FLTrust_RW', 'FLTrust_FairG', 'FLTrust_RW_FairG', 'FLTrust_RW_FairCosG'] and deltas is not None and TS_ratio is not None:
            if self.fairg and self.algorithm in ['FLTrust_FairG', 'FLTrust_RW_FairG']:
                # 使用FairG进行客户端筛选
                selected_client_ids = self.fairg.filter_clients(client_discrimination_scores, tau=HYPERPARAMETERS['KMEANS_TAU_RATIO'])

                # 打印筛选结果
                num_selected = len(selected_client_ids)
                total_clients = len(deltas)
                print(f"筛选后参与聚合的客户端数量: {num_selected} / {total_clients}")

                excluded_clients = [cid for cid in deltas.keys() if cid not in selected_client_ids]
                if excluded_clients:
                    excluded_str = ', '.join(map(str, excluded_clients))
                    print(f"未参与聚合的客户端编号: {excluded_str}")
                else:
                    print("所有客户端均参与聚合。")

                if num_selected == 0:
                    print("没有客户端通过FairG筛选，跳过本轮聚合。")
                    return

                # 仅聚合通过筛选的客户端
                filtered_deltas = {client_id: deltas[client_id] for client_id in selected_client_ids}
                filtered_TS_ratio = {client_id: TS_ratio[client_id] for client_id in selected_client_ids}
                
                # 重新归一化TS_ratio
                total_filtered_TS = sum(filtered_TS_ratio.values()) + 1e-10
                filtered_TS_ratio = {client_id: ts / total_filtered_TS for client_id, ts in filtered_TS_ratio.items()}

                aggregated_weights = self._fltrust_aggregate(filtered_deltas, filtered_TS_ratio)
                
            elif self.faircosg and self.algorithm in ['FLTrust_RW_FairCosG']:
                # 使用FairCosG进行客户端筛选
                if excluded_clients_info is not None:
                    selected_client_ids, faircosg_scores = excluded_clients_info
                else:
                    selected_client_ids, faircosg_scores = self.faircosg.filter_clients(self.global_model, self.clients, MLP)

                # 仅聚合通过筛选的客户端
                filtered_deltas = {client_id: deltas[client_id] for client_id in selected_client_ids}
                filtered_TS_ratio = {client_id: TS_ratio[client_id] for client_id in selected_client_ids}

                if len(filtered_deltas) == 0:
                    print("没有客户端通过FairCosG筛选，跳过本轮聚合。")
                    return

                # 重新归一化TS_ratio
                total_filtered_TS = sum(filtered_TS_ratio.values()) + 1e-10
                filtered_TS_ratio = {client_id: ts / total_filtered_TS for client_id, ts in filtered_TS_ratio.items()}

                aggregated_weights = self._fltrust_aggregate(filtered_deltas, filtered_TS_ratio)
            else:
                aggregated_weights = self._fltrust_aggregate(deltas, TS_ratio)
        else:
            raise ValueError(f"未知或不完整的算法参数: {self.algorithm}")

        self.global_model.load_state_dict(aggregated_weights)  # 更新全局模型权重

    def _fltrust_aggregate(self, deltas, TS_ratio):
        """
        FLTrust算法的聚合方法，根据相似度加权客户端的delta。
        """
        fltrust_alpha = self.hyperparams['FLTRUST_ALPHA']

        # 计算 global_delta = SUM(每个端的Delta_norm * TS_ratio)
        global_delta = {}
        for key in self.global_model.state_dict().keys():
            global_delta[key] = torch.zeros_like(self.global_model.state_dict()[key]).to(DEVICE)
            for client_id, delta in deltas.items():
                weight = TS_ratio.get(client_id, 0.0)
                global_delta[key] += delta[key] * weight

        # 更新全局模型
        new_global_state_dict = {}
        for key in self.global_model.state_dict().keys():
            new_global_state_dict[key] = self.global_model.state_dict()[key] + fltrust_alpha * global_delta[key]
        return new_global_state_dict

    def run_round(self, round_num, test_df, y_test_values, model_class):
        """
        运行一轮全局训练，根据算法不同调用不同的聚合方法。
        """
        client_deltas = {}
        client_discrimination_scores = {}
        faircosg_scores = {}
        excluded_clients_info = None
        local_losses = []

        if self.algorithm in ['FLTrust', 'FLTrust_RW', 'FLTrust_FairG', 'FLTrust_RW_FairG', 'FLTrust_RW_FairCosG']:
            # FLTrust逻辑
            for client in self.clients:
                delta, loss = client.local_train_fltrust(self.global_model, self.global_model.state_dict(), self.hyperparams['FLTRUST_ALPHA'], self.hyperparams['FLTRUST_BETA'])
                client_deltas[client.client_id] = delta
                local_losses.append(loss)

            # 计算每个客户端的歧视分数（如果使用FairG）
            if self.fairg and self.algorithm in ['FLTrust_FairG', 'FLTrust_RW_FairG']:
                for client in self.clients:
                    dis_score = self.fairg.compute_discrimination_score(client.model, DEVICE)
                    client_discrimination_scores[client.client_id] = dis_score

            # 计算FairCosG分数（如果使用FairCosG）
            if self.faircosg and self.algorithm in ['FLTrust_RW_FairCosG']:
                excluded_clients_info = self.faircosg.filter_clients(self.global_model, self.clients, model_class)
                faircosg_scores = excluded_clients_info[1]

            # 服务器在其数据集上训练指定的epoch数
            if self.server_data is not None:
                self.server_model.load_state_dict(self.global_model.state_dict())
                self.server_model.train()
                
                # 对服务器数据进行标准化
                server_data_copy = self.server_data.copy()
                server_data_copy[numerical_columns] = scaler.transform(server_data_copy[numerical_columns])
                
                server_X_tensor = torch.tensor(server_data_copy.drop(['income', 'sex'], axis=1).values, dtype=torch.float32).to(HYPERPARAMETERS['DEVICE'])
                server_y_tensor = torch.tensor(server_data_copy['income'].values, dtype=torch.long).to(HYPERPARAMETERS['DEVICE'])
                server_dataset = TensorDataset(server_X_tensor, server_y_tensor)
                server_loader = DataLoader(
                    server_dataset,
                    batch_size=HYPERPARAMETERS['BATCH_SIZE'],
                    shuffle=True
                )
                for epoch in range(self.hyperparams['SERVER_EPOCHS']):  # 使用超参数
                    for batch in server_loader:
                        X_batch, y_batch = batch
                        X_batch, y_batch = X_batch.to(HYPERPARAMETERS['DEVICE']), y_batch.to(HYPERPARAMETERS['DEVICE'])
                        self.server_optimizer.zero_grad()
                        logits = self.server_model(X_batch)
                        loss = self.server_criterion(logits, y_batch)
                        loss.backward()
                        self.server_optimizer.step()

                # 计算 root_delta = w_server - w_global
                root_delta = {}
                server_state_dict = self.server_model.state_dict()
                global_state_dict = self.global_model.state_dict()
                for key in global_state_dict.keys():
                    root_delta[key] = server_state_dict[key] - global_state_dict[key]

                # 将 root_delta 转换为扁平向量
                root_weight_flat = torch.cat([v.flatten() for v in root_delta.values()]).cpu().numpy()
                root_norm = np.linalg.norm(root_weight_flat) + 1e-10  # 防止除零

            else:
                raise ValueError("FLTrust requires server_data to compute root_norm.")

            # 计算每个客户端的相似度权重TS并归一化 Delta
            similarity_scores = {}
            deltas_norm = {}
            for client_id, delta in client_deltas.items():
                # 将 delta 转换为扁平向量
                delta_flat = torch.cat([v.flatten() for v in delta.values()]).cpu().numpy()
                delta_norm = np.linalg.norm(delta_flat) + 1e-10  # 防止除零

                # 归一化 delta
                scaling_factor = root_norm / delta_norm
                deltas_norm[client_id] = {k: (v * scaling_factor) for k, v in delta.items()}

                # 计算余弦相似度
                cosine_sim = np.dot(delta_flat, root_weight_flat) / (delta_norm * root_norm)
                cosine_sim = max(cosine_sim, 0)  # 若相似度小于0，则设为0
                similarity_scores[client_id] = cosine_sim

            # 计算总相似度
            total_TS = sum(similarity_scores.values()) + 1e-10  # 防止除零

            # 计算TS_ratio
            TS_ratio = {client_id: sim / total_TS for client_id, sim in similarity_scores.items()}

            # 通过deltas_norm和TS_ratio得到globel_delta从而得到新的全局模型
            self.aggregate(deltas=deltas_norm, TS_ratio=TS_ratio, client_discrimination_scores=client_discrimination_scores, faircosg_scores=faircosg_scores, excluded_clients_info=excluded_clients_info)

            # 进行全局测试
            loss_test, accuracy, fairness_metrics, per_category = test_inference_modified(
                self.global_model,
                test_loader,
                model_class
            )

            return accuracy, fairness_metrics['EOD'], fairness_metrics['SPD'], per_category, client_discrimination_scores, faircosg_scores

        else:
            raise ValueError(f"未知的算法: {self.algorithm}")