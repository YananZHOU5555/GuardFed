# FairCosG.py - 修正版FairCos客户端筛选框架

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import copy
import math

# 注意：导入路径已更新为相对导入
# from HYPERPARAMETERS import DEVICE, HYPERPARAMETERS, SEED, algorithms, attack_forms, MALICIOUS_CLIENTS
# from data import ...
# from function import compute_fairness_metrics

class FairCosG:
    def __init__(self, server_data=None, lambda_param=1.0):
        """
        初始化FairCosG框架
        """
        self.server_data = server_data
        self.lambda_param = lambda_param  # 可以动态设置lambda参数
        self.eod_tolerance = HYPERPARAMETERS['FAIRCOSG_EOD_TOLERANCE'] 
        self.score_threshold = 0.2  # 改为0.2
        self.faircosg_alpha = HYPERPARAMETERS['FAIRCOSG_ALPHA']
        self.faircosg_beta = HYPERPARAMETERS['FAIRCOSG_BETA']
        self.server_epochs = HYPERPARAMETERS['FAIRCOSG_SERVER_EPOCHS']
        
        # 服务器模型（每轮都会重新训练）
        self.server_model = None

    def train_server_model(self, global_model, model_class, reweighing_weights):
        """
        1. 创建服务器模型并训练，对服务器数据进行标准化
        服务器有IID数据集（10%总数据量），也应用RW权重保证公平性
        """
        if self.server_data is None:
            raise ValueError("FairCosG requires server_data")
        
        # 创建服务器模型并加载全局模型权重
        self.server_model = model_class(HYPERPARAMETERS['INPUT_SIZE'], HYPERPARAMETERS['OUTPUT_SIZE']).to(HYPERPARAMETERS['DEVICE'])
        self.server_model.load_state_dict(global_model.state_dict())
        self.server_model.train()
        
        server_optimizer = optim.Adam(self.server_model.parameters(), lr=self.faircosg_beta)
        server_criterion = nn.CrossEntropyLoss(reduction='none')  # 用于应用样本权重
        
        # 对服务器数据进行标准化
        server_data_copy = self.server_data.copy()
        server_data_copy[numerical_columns] = scaler.transform(server_data_copy[numerical_columns])
        
        server_X_tensor = torch.tensor(server_data_copy.drop('income', axis=1).values, dtype=torch.float32).to(HYPERPARAMETERS['DEVICE'])
        server_y_tensor = torch.tensor(server_data_copy['income'].values, dtype=torch.long).to(HYPERPARAMETERS['DEVICE'])
        server_sex_tensor = torch.tensor(server_data_copy['sex'].values, dtype=torch.long).to(HYPERPARAMETERS['DEVICE'])
        
        # 计算服务器数据的RW权重
        server_sample_weights = []
        for i in range(len(server_data_copy)):
            sex_val = server_sex_tensor[i].item()
            income_val = server_y_tensor[i].item()
            weight = reweighing_weights.get((sex_val, income_val), 1.0)
            server_sample_weights.append(weight)
        
        server_weights_tensor = torch.tensor(server_sample_weights, dtype=torch.float32).to(HYPERPARAMETERS['DEVICE'])
        
        server_dataset = TensorDataset(server_X_tensor, server_y_tensor, server_sex_tensor, server_weights_tensor)
        server_loader = DataLoader(
            server_dataset,
            batch_size=HYPERPARAMETERS['BATCH_SIZE'],
            shuffle=True
        )
        
        # 服务器训练一轮得到server模型m0（应用RW权重）
        total_loss = 0.0
        total_batches = 0
        
        for epoch in range(self.server_epochs):
            for batch in server_loader:
                X_batch, y_batch, _, weights_batch = batch
                X_batch, y_batch, weights_batch = X_batch.to(HYPERPARAMETERS['DEVICE']), y_batch.to(HYPERPARAMETERS['DEVICE']), weights_batch.to(HYPERPARAMETERS['DEVICE'])
                
                server_optimizer.zero_grad()
                logits = self.server_model(X_batch)
                loss = server_criterion(logits, y_batch)  # [batch_size]
                
                # 应用RW样本权重
                loss = loss * weights_batch  # 应用样本权重
                loss = loss.sum() / weights_batch.sum()  # 加权平均
                
                loss.backward()
                server_optimizer.step()
                total_loss += loss.item()
                total_batches += 1
        
        avg_loss = total_loss / total_batches if total_batches > 0 else 0.0
        print(f"服务器模型m0训练完成（应用RW权重），数据集大小: {len(self.server_data)}, 平均损失: {avg_loss:.6f}")

    def compute_cosine_similarity(self, client_model):
        """
        计算客户端模型mi和服务器模型m0的余弦相似度
        直接比较模型权重（转化为向量并归一化）
        """
        if self.server_model is None:
            raise ValueError("Server model not trained yet")
        
        # 将客户端模型权重转化为向量（需要分离梯度）
        client_weights_flat = torch.cat([param.flatten().detach() for param in client_model.parameters()]).cpu().numpy()
        client_norm = np.linalg.norm(client_weights_flat) + 1e-10  # 防止除零
        
        # 将服务器模型权重转化为向量（需要分离梯度）
        server_weights_flat = torch.cat([param.flatten().detach() for param in self.server_model.parameters()]).cpu().numpy()
        server_norm = np.linalg.norm(server_weights_flat) + 1e-10  # 防止除零
        
        # 计算余弦相似度
        cosine_sim = np.dot(client_weights_flat, server_weights_flat) / (client_norm * server_norm)
        cosine_sim = max(cosine_sim, 0)  # 若相似度小于0，则设为0
        
        return cosine_sim

    def compute_eod_on_server_data(self, client_model):
        """计算客户端模型在服务器数据上的EOD"""
        if self.server_data is None:
            return 0.0
        
        client_model.eval()
        
        # 准备服务器数据 - 重要：需要标准化！
        server_data_copy = self.server_data.copy()
        server_data_copy[numerical_columns] = scaler.transform(server_data_copy[numerical_columns])
        
        server_X_tensor = torch.tensor(server_data_copy.drop('income', axis=1).values, dtype=torch.float32).to(HYPERPARAMETERS['DEVICE'])
        server_y_tensor = torch.tensor(server_data_copy['income'].values, dtype=torch.long).to(HYPERPARAMETERS['DEVICE'])
        server_sex_tensor = torch.tensor(server_data_copy['sex'].values, dtype=torch.long).to(HYPERPARAMETERS['DEVICE'])
        
        # 模型推理
        with torch.no_grad():
            logits = client_model(server_X_tensor)
            preds = torch.argmax(logits, dim=1)
            y_pred = preds.cpu().numpy()
            y_true = server_y_tensor.cpu().numpy()
            sex_values = server_sex_tensor.cpu().numpy()
        
        # 计算EOD
        fairness_metrics = compute_fairness_metrics(y_true, y_pred, sex_values)
        return abs(fairness_metrics["EOD"])

    def compute_faircos_score(self, cosine_similarity, eod):
        """根据余弦相似度和EOD计算FairCos分数"""
        cos = max(0, cosine_similarity)  # ReLU确保非负
        
        # 对小的EOD值提供容忍度，避免过度惩罚正常的训练波动
        adjusted_eod = max(0, abs(eod) - self.eod_tolerance)
        
        # FairCos分数设计：余弦相似度 * exp(-λ * adjusted_eod)
        score = cos * math.exp(-self.lambda_param * adjusted_eod)
        
        return score

    def filter_clients_after_training(self, global_model, clients, model_class, reweighing_weights):
        """
        在客户端完成训练后，使用FairCos分数筛选客户端
        
        正确的流程：
        1. 服务器训练得到m0（应用RW权重）
        2. 客户端已经完成训练得到mi
        3. 计算mi和m0的余弦相似度
        4. 计算每个客户端在服务器数据上的EOD
        5. 计算FairCos分数
        6. 筛选分数>0.1的参与聚合
        """
        
        # 1. 训练服务器模型m0（应用RW权重）
        self.train_server_model(global_model, model_class, reweighing_weights)
        
        # 2. 对每个客户端计算FairCos分数
        faircos_scores = {}
        cosine_similarities = {}
        eods = {}
        
        print(f"\nFairCosG客户端筛选详情 (λ={self.lambda_param}):")
        print(f"{'客户端':<8s} {'类型':<8s} {'Cosine':<8s} {'EOD':<10s} {'FairCos分数':<12s} {'状态':<8s}")
        print("-" * 65)
        
        for client in clients:
            # 计算余弦相似度
            cosine_sim = self.compute_cosine_similarity(client.model)
            cosine_similarities[client.client_id] = cosine_sim
            
            # 计算EOD
            client_eod = self.compute_eod_on_server_data(client.model)
            eods[client.client_id] = client_eod
            
            # 计算FairCos分数
            faircos_score = self.compute_faircos_score(cosine_sim, client_eod)
            faircos_scores[client.client_id] = faircos_score
            
            # 判断是否参与聚合
            client_type = "恶意" if client.client_id in MALICIOUS_CLIENTS else "正常"
            status = "参与" if faircos_score > self.score_threshold else "过滤"
            
            print(f"{client.client_id:<8d} {client_type:<8s} {cosine_sim:<8.4f} {client_eod:<10.6f} {faircos_score:<12.6f} {status:<8s}")
        
        # 3. 筛选参与聚合的客户端
        filtered_client_ids = [cid for cid, score in faircos_scores.items() if score > self.score_threshold]
        excluded_clients = [cid for cid, score in faircos_scores.items() if score <= self.score_threshold]
        
        # 统计筛选结果
        num_selected = len(filtered_client_ids)
        total_clients = len(faircos_scores)
        print(f"\n筛选后参与聚合的客户端数量: {num_selected} / {total_clients}")
        
        if excluded_clients:
            excluded_malicious = [cid for cid in excluded_clients if cid in MALICIOUS_CLIENTS]
            excluded_normal = [cid for cid in excluded_clients if cid not in MALICIOUS_CLIENTS]
            excluded_str = ', '.join(map(str, excluded_clients))
            print(f"未参与聚合的客户端编号: {excluded_str}")
            print(f"其中恶意客户端: {excluded_malicious}, 正常客户端: {excluded_normal}")
        else:
            print("所有客户端均参与聚合。")
        
        return filtered_client_ids, faircos_scores

    def get_faircos_scores_for_tracking(self, faircos_scores):
        """
        获取用于跟踪分析的详细FairCos分数信息
        """
        # 计算恶意客户端和正常客户端的平均分数
        malicious_scores = [faircos_scores[cid] for cid in MALICIOUS_CLIENTS if cid in faircos_scores]
        normal_scores = [faircos_scores[cid] for cid in faircos_scores.keys() if cid not in MALICIOUS_CLIENTS]
        
        avg_malicious = np.mean(malicious_scores) if malicious_scores else 0
        avg_normal = np.mean(normal_scores) if normal_scores else 0
        
        stats = {
            'faircos_scores': faircos_scores,
            'avg_malicious_score': avg_malicious,
            'avg_normal_score': avg_normal,
            'malicious_scores': malicious_scores,
            'normal_scores': normal_scores
        }
        
        return stats