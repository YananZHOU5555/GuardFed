# src/data_loader.py - 统一数据加载器，支持多个数据集

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings

warnings.filterwarnings('ignore')


class DatasetLoader:
    """统一的数据集加载器，支持Adult和COMPAS数据集"""

    def __init__(self, dataset_name='adult', seed=123, device='cpu'):
        """
        初始化数据加载器

        参数:
            dataset_name: 数据集名称 ('adult' 或 'compas')
            seed: 随机种子
            device: 设备 ('cpu' 或 'cuda')
        """
        self.dataset_name = dataset_name.lower()
        self.seed = seed
        self.device = torch.device(device)

        # 设置随机种子
        np.random.seed(seed)
        torch.manual_seed(seed)

        # 数据集路径
        self.data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')

        # 初始化变量
        self.train_df = None
        self.test_df = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.sex_train = None
        self.sex_test = None
        self.scaler = None
        self.label_encoders = {}
        self.categorical_columns = []
        self.numerical_columns = []

        # 敏感属性配置
        self.sensitive_column = None
        self.A_PRIVILEGED = None
        self.A_UNPRIVILEGED = None

        # 加载数据
        self._load_dataset()

    def _load_dataset(self):
        """根据数据集名称加载相应的数据"""
        if self.dataset_name == 'adult':
            self._load_adult()
        elif self.dataset_name == 'compas':
            self._load_compas()
        else:
            raise ValueError(f"不支持的数据集: {self.dataset_name}")

    def _load_adult(self):
        """加载Adult Income数据集"""
        print(f"\n{'='*60}")
        print("加载 Adult Income 数据集")
        print(f"{'='*60}")

        # 数据路径
        train_path = os.path.join(self.data_dir, 'adult', 'adult.data')
        test_path = os.path.join(self.data_dir, 'adult', 'adult.test')

        # 列名
        columns = [
            'age', 'workclass', 'fnlwgt', 'education', 'education-num',
            'marital-status', 'occupation', 'relationship', 'race', 'sex',
            'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
        ]

        # 读取数据
        train_df = pd.read_csv(train_path, names=columns, sep=r'\s*,\s*',
                               engine='python', na_values='?', skiprows=0)
        test_df = pd.read_csv(test_path, names=columns, sep=r'\s*,\s*',
                              engine='python', na_values='?', skiprows=1)

        # 清理income列（测试集有多余的'.'）
        test_df['income'] = test_df['income'].str.rstrip('.')

        # 删除缺失值
        train_df = train_df.dropna()
        test_df = test_df.dropna()

        # 二值化income
        train_df['income'] = (train_df['income'] == '>50K').astype(int)
        test_df['income'] = (test_df['income'] == '>50K').astype(int)

        # 二值化sex
        train_df['sex'] = (train_df['sex'] == 'Male').astype(int)
        test_df['sex'] = (test_df['sex'] == 'Male').astype(int)

        # 定义特征类型
        self.categorical_columns = [
            'workclass', 'education', 'marital-status', 'occupation',
            'relationship', 'race', 'native-country'
        ]
        self.numerical_columns = [
            'age', 'fnlwgt', 'education-num', 'capital-gain',
            'capital-loss', 'hours-per-week'
        ]

        # 类别编码
        for col in self.categorical_columns:
            le = LabelEncoder()
            train_df[col] = le.fit_transform(train_df[col])
            test_df[col] = le.transform(test_df[col])
            self.label_encoders[col] = le

        # 标准化
        self.scaler = StandardScaler()
        train_df[self.numerical_columns] = self.scaler.fit_transform(
            train_df[self.numerical_columns]
        )
        test_df[self.numerical_columns] = self.scaler.transform(
            test_df[self.numerical_columns]
        )

        # 分离特征和标签
        self.train_df = train_df
        self.test_df = test_df
        self.X_train = train_df.drop(['income', 'sex'], axis=1)
        self.y_train = train_df['income']
        self.sex_train = train_df['sex'].values
        self.X_test = test_df.drop(['income', 'sex'], axis=1)
        self.y_test = test_df['income']
        self.sex_test = test_df['sex'].values

        # 敏感属性配置
        self.sensitive_column = 'sex'
        self.A_PRIVILEGED = 1  # Male
        self.A_UNPRIVILEGED = 0  # Female

        print(f"[OK] Adult数据集加载完成")
        print(f"   训练集: {len(self.train_df)} 样本")
        print(f"   测试集: {len(self.test_df)} 样本")
        print(f"   特征数: {self.X_train.shape[1]}")
        print(f"   敏感属性: {self.sensitive_column} (Male=1, Female=0)")

    def _load_compas(self):
        """加载COMPAS数据集"""
        print(f"\n{'='*60}")
        print("加载 COMPAS 数据集")
        print(f"{'='*60}")

        # 数据路径
        data_path = os.path.join(self.data_dir, 'compas', 'compas-scores-two-years.csv')

        # 读取数据
        df = pd.read_csv(data_path)

        # 数据清洗（参考ProPublica的分析）
        df = df[
            (df['days_b_screening_arrest'] <= 30) &
            (df['days_b_screening_arrest'] >= -30) &
            (df['is_recid'] != -1) &
            (df['c_charge_degree'] != 'O') &
            (df['score_text'] != 'N/A')
        ]

        # 选择特征
        features_to_keep = [
            'sex', 'age', 'age_cat', 'race',
            'juv_fel_count', 'juv_misd_count', 'juv_other_count',
            'priors_count', 'c_charge_degree', 'two_year_recid'
        ]
        df = df[features_to_keep].copy()

        # 删除缺失值
        df = df.dropna()

        # 二值化race（African-American vs Others）
        df['race'] = (df['race'] == 'African-American').astype(int)

        # 二值化sex
        df['sex'] = (df['sex'] == 'Male').astype(int)

        # 二值化age_cat
        age_cat_mapping = {'Less than 25': 0, '25 - 45': 1, 'Greater than 45': 2}
        df['age_cat'] = df['age_cat'].map(age_cat_mapping)

        # 二值化c_charge_degree
        df['c_charge_degree'] = (df['c_charge_degree'] == 'F').astype(int)

        # 定义特征类型
        self.categorical_columns = ['age_cat', 'c_charge_degree']
        self.numerical_columns = [
            'age', 'juv_fel_count', 'juv_misd_count',
            'juv_other_count', 'priors_count'
        ]

        # 标准化
        self.scaler = StandardScaler()
        df[self.numerical_columns] = self.scaler.fit_transform(
            df[self.numerical_columns]
        )

        # 划分训练集和测试集
        train_df, test_df = train_test_split(
            df, test_size=0.3, random_state=self.seed, stratify=df['race']
        )

        # 分离特征和标签
        self.train_df = train_df.reset_index(drop=True)
        self.test_df = test_df.reset_index(drop=True)
        self.X_train = train_df.drop(['two_year_recid', 'race'], axis=1)
        self.y_train = train_df['two_year_recid']
        self.sex_train = train_df['race'].values
        self.X_test = test_df.drop(['two_year_recid', 'race'], axis=1)
        self.y_test = test_df['two_year_recid']
        self.sex_test = test_df['race'].values

        # 敏感属性配置
        self.sensitive_column = 'race'
        self.A_PRIVILEGED = 1  # African-American
        self.A_UNPRIVILEGED = 0  # Others

        print(f"[OK] COMPAS数据集加载完成")
        print(f"   训练集: {len(self.train_df)} 样本")
        print(f"   测试集: {len(self.test_df)} 样本")
        print(f"   特征数: {self.X_train.shape[1]}")
        print(f"   敏感属性: {self.sensitive_column} (African-American=1, Others=0)")

    def get_tensors(self):
        """返回PyTorch张量格式的数据"""
        X_train_tensor = torch.tensor(self.X_train.values, dtype=torch.float32).to(self.device)
        y_train_tensor = torch.tensor(self.y_train.values, dtype=torch.long).to(self.device)
        X_test_tensor = torch.tensor(self.X_test.values, dtype=torch.float32).to(self.device)
        y_test_tensor = torch.tensor(self.y_test.values, dtype=torch.long).to(self.device)
        sex_test_tensor = torch.tensor(self.sex_test, dtype=torch.long).to(self.device)

        return {
            'X_train': X_train_tensor,
            'y_train': y_train_tensor,
            'X_test': X_test_tensor,
            'y_test': y_test_tensor,
            'sex_train': self.sex_train,
            'sex_test': sex_test_tensor
        }

    def create_test_loader(self, batch_size=256):
        """创建测试数据加载器"""
        tensors = self.get_tensors()
        test_dataset = TensorDataset(
            tensors['X_test'],
            tensors['y_test'],
            tensors['sex_test']
        )
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        return test_loader

    def split_server_client_data(self, server_ratio=0.1):
        """
        分割服务器和客户端数据

        参数:
            server_ratio: 服务器数据比例

        返回:
            server_df: 服务器数据（IID，分层采样）
            client_df: 客户端数据
        """
        # 分层采样保证服务器数据的IID性
        server_df = self.train_df.groupby(self.sensitive_column, group_keys=False).apply(
            lambda x: x.sample(frac=server_ratio, random_state=self.seed)
        )
        client_df = self.train_df.drop(server_df.index)

        return server_df.reset_index(drop=True), client_df.reset_index(drop=True)

    def create_client_data_dict(self, client_df, num_clients=20, alpha=1.0):
        """
        使用Dirichlet分布创建Non-IID客户端数据字典

        参数:
            client_df: 客户端数据
            num_clients: 客户端数量
            alpha: Dirichlet浓度参数（越小越Non-IID）

        返回:
            client_data_dict: {client_id: {'X': tensor, 'y': tensor, 'sensitive': array}}
        """
        client_data_dict = {i: {"X": [], "y": [], "sensitive": []} for i in range(num_clients)}

        # 分别对特权群体和非特权群体进行Dirichlet划分
        for group_val in [self.A_PRIVILEGED, self.A_UNPRIVILEGED]:
            group_df = client_df[client_df[self.sensitive_column] == group_val]
            group_indices = group_df.index.tolist()

            # Dirichlet采样
            proportions = np.random.dirichlet([alpha] * num_clients)
            proportions = (np.cumsum(proportions) * len(group_indices)).astype(int)[:-1]

            # 分配数据
            splits = np.split(group_indices, proportions)
            for i in range(num_clients):
                if len(splits[i]) > 0:
                    client_subset = client_df.loc[splits[i]]
                    X_subset = client_subset.drop(['income' if self.dataset_name == 'adult' else 'two_year_recid',
                                                   self.sensitive_column], axis=1)
                    y_subset = client_subset['income' if self.dataset_name == 'adult' else 'two_year_recid']

                    client_data_dict[i]["X"].append(torch.tensor(X_subset.values, dtype=torch.float32))
                    client_data_dict[i]["y"].append(torch.tensor(y_subset.values, dtype=torch.long))
                    client_data_dict[i]["sensitive"].append(np.full(len(splits[i]), group_val))

        # 合并每个客户端的数据
        for i in range(num_clients):
            if len(client_data_dict[i]["X"]) > 0:
                client_data_dict[i]["X"] = torch.cat(client_data_dict[i]["X"]).to(self.device)
                client_data_dict[i]["y"] = torch.cat(client_data_dict[i]["y"]).to(self.device)
                client_data_dict[i]["sensitive"] = np.concatenate(client_data_dict[i]["sensitive"])
            else:
                # 空客户端
                client_data_dict[i]["X"] = torch.empty(0, self.X_train.shape[1], dtype=torch.float32).to(self.device)
                client_data_dict[i]["y"] = torch.empty(0, dtype=torch.long).to(self.device)
                client_data_dict[i]["sensitive"] = np.array([], dtype=int)

        return client_data_dict

    def get_info(self):
        """返回数据集信息"""
        return {
            'dataset_name': self.dataset_name,
            'num_features': self.X_train.shape[1],
            'num_train': len(self.train_df),
            'num_test': len(self.test_df),
            'sensitive_column': self.sensitive_column,
            'A_PRIVILEGED': self.A_PRIVILEGED,
            'A_UNPRIVILEGED': self.A_UNPRIVILEGED,
            'categorical_columns': self.categorical_columns,
            'numerical_columns': self.numerical_columns
        }


# 测试代码
if __name__ == "__main__":
    print("\n" + "="*80)
    print("测试 Adult 数据集加载")
    print("="*80)

    loader_adult = DatasetLoader(dataset_name='adult', seed=123)
    info_adult = loader_adult.get_info()
    print(f"\n数据集信息:")
    for key, value in info_adult.items():
        print(f"  {key}: {value}")

    print("\n" + "="*80)
    print("测试 COMPAS 数据集加载")
    print("="*80)

    loader_compas = DatasetLoader(dataset_name='compas', seed=123)
    info_compas = loader_compas.get_info()
    print(f"\n数据集信息:")
    for key, value in info_compas.items():
        print(f"  {key}: {value}")

    print("\n[OK] 数据加载器测试完成！")
