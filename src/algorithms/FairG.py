# FairG.py

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
# 注意：导入路径已更新
# from HYPERPARAMETERS import HYPERPARAMETERS, SEED

class FairG:
    def __init__(self, R=500):
        """
        初始化FairG框架，生成G_data, G_data_A, G_data_B。
        """
        self.R = R
        np.random.seed(SEED)
        self.generate_G_data()
        self.create_G_data_A_B()
        self.check_G_data_A_B()

    def generate_G_data(self):
        """
        生成G_data数据集。
        """
        self.G_data = pd.DataFrame({
            'age': np.random.uniform(-1.6322, 3.9257, self.R),
            'workclass': np.random.randint(0, 7, self.R),
            'fnlwgt': np.random.uniform(-1.6661, 12.2565, self.R),
            'education': np.random.randint(0, 16, self.R),
            'education-num': np.random.uniform(-3.5771, 2.3054, self.R),
            'marital-status': np.random.randint(0, 7, self.R),
            'occupation': np.random.randint(0, 14, self.R),
            'relationship': np.random.randint(0, 6, self.R),
            'race': np.random.randint(0, 5, self.R),
            'capital-gain': np.random.uniform(-0.1474, 13.3546, self.R),
            'capital-loss': np.random.uniform(-0.2186, 10.5558, self.R),
            'hours-per-week': np.random.uniform(-3.3332, 4.8472, self.R),
            'native-country': np.random.randint(0, 41, self.R)
        })

    def create_G_data_A_B(self):
        """
        生成G_data_A和G_data_B数据集。
        """
        self.G_data_A = self.G_data.copy()
        self.G_data_A['sex'] = 1  # 全部男性

        self.G_data_B = self.G_data.copy()
        self.G_data_B['sex'] = 0  # 全部女性

    def check_G_data_A_B(self):
        """
        检查G_data_A和G_data_B是否正确生成。
        """
        assert list(self.G_data.columns) == list(self.G_data_A.drop('sex', axis=1).columns), "G_data_A的特征与G_data不匹配！"
        assert list(self.G_data.columns) == list(self.G_data_B.drop('sex', axis=1).columns), "G_data_B的特征与G_data不匹配！"
        print("G_data_A和G_data_B已成功生成并通过检查。")

    def compute_discrimination_score(self, model, device):
        """
        计算模型在G_data_A和G_data_B上的歧视分数。
        """
        model.eval()
        with torch.no_grad():
            # 转换为Tensor并移动到设备
            X_A = torch.tensor(self.G_data_A.values, dtype=torch.float32).to(device)
            X_B = torch.tensor(self.G_data_B.values, dtype=torch.float32).to(device)

            # 模型预测
            logits_A = model(X_A)
            probs_A = torch.softmax(logits_A, dim=1)[:, 1]  # 类别1的概率

            logits_B = model(X_B)
            probs_B = torch.softmax(logits_B, dim=1)[:, 1]  # 类别1的概率

        # 计算L1范数（总绝对差）
        discrimination_score = torch.sum(torch.abs(probs_A - probs_B)).item()
        return discrimination_score

    def filter_clients(self, client_scores, tau):
        """
        使用K-Means聚类筛选歧视分数低的客户端，同时打印每个客户端的可疑分数，
        筛选后参与聚合的客户端数量以及未参与聚合的客户端编号。
        """
        if len(client_scores) == 0:
            return []

        # 打印每个客户端的可疑分数
        print("每个客户端的可疑分数:")
        for client_id, score in client_scores.items():
            print(f"客户端 {client_id}: 可疑分数 = {score:.4f}")

        # 获取超参数
        n_clusters = HYPERPARAMETERS.get('KMEANS_N_CLUSTERS', 2)
        init_method = HYPERPARAMETERS.get('KMEANS_INIT', 'k-means++')
        random_state = HYPERPARAMETERS.get('KMEANS_RANDOM_STATE', SEED)
        max_iter = HYPERPARAMETERS.get('KMEANS_MAX_ITER', 300)
        n_init = HYPERPARAMETERS.get('KMEANS_N_INIT', 10)
        standardize = HYPERPARAMETERS.get('KMEANS_STANDARDIZE', True)
        tau_ratio = HYPERPARAMETERS.get('KMEANS_TAU_RATIO', 0.05)

        # 准备数据
        scores = np.array(list(client_scores.values())).reshape(-1, 1)

        # 标准化处理
        if standardize:
            scaler = StandardScaler()
            scores = scaler.fit_transform(scores)

        # K-Means聚类
        kmeans = KMeans(n_clusters=n_clusters, init=init_method, random_state=random_state, max_iter=max_iter, n_init=n_init)
        labels = kmeans.fit_predict(scores)

        # 计算每个簇的中心
        cluster_centers = kmeans.cluster_centers_.flatten()

        # 确定哪个簇是高歧视分数的簇（中心值最大的簇）
        high_discrimination_label = np.argmax(cluster_centers)

        # 获取低歧视分数的客户端索引
        low_discrimination_indices = [idx for idx, label in enumerate(labels) if label != high_discrimination_label]

        # 计算τ阈值
        tau_threshold = tau_ratio * self.R

        # 判断高歧视分数簇的中心是否超过阈值
        if cluster_centers[high_discrimination_label] > tau_threshold:
            selected_clients = low_discrimination_indices
            excluded_clients = [idx for idx in range(len(client_scores)) if idx not in selected_clients]
        else:
            selected_clients = list(range(len(client_scores)))  # 所有客户端都参与聚合
            excluded_clients = []

        # 打印筛选结果
        num_selected = len(selected_clients)
        total_clients = len(client_scores)
        print(f"筛选后参与聚合的客户端数量: {num_selected} / {total_clients}")

        if excluded_clients:
            excluded_str = ', '.join(map(str, excluded_clients))
            print(f"未参与聚合的客户端编号: {excluded_str}")
        else:
            print("所有客户端均参与聚合。")

        return selected_clients