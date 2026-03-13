# GuardFed - 公平性感知的拜占庭鲁棒联邦学习框架

**版本**: v2.0.0  
**日期**: 2026-02-02  
**状态**: 🟡 结构重组完成 (70%)

---

## 📋 项目概述

GuardFed 是一个联邦学习研究框架，专注于在拜占庭攻击环境下实现公平性感知的鲁棒聚合算法。

### 核心特性

- ✅ **6个联邦学习算法**: FedAvg, FLTrust, FairFed, Medium, FairG, FairCosG
- ✅ **2个防御框架**: FairG (歧视分数筛选), FairCosG (余弦相似度+公平性)
- ✅ **6种攻击模型**: 公平性攻击、准确性攻击、混合攻击
- ✅ **2个数据集**: Adult Income, COMPAS
- ✅ **统一数据加载器**: 一行代码切换数据集

---

## 📁 项目结构

```
GuardFed/
├── data/                        # 数据集
│   ├── adult/                   # Adult Income数据集
│   └── compas/                  # COMPAS数据集
│
├── src/                         # 源代码
│   ├── algorithms/              # 联邦学习算法
│   │   ├── FedAvg.py           # FedAvg + Reweighting
│   │   ├── FLTrust.py          # FLTrust (信任评分)
│   │   ├── FairFed.py          # FairFed (公平性感知)
│   │   ├── Medium.py           # Median聚合
│   │   ├── FairG.py            # FairG (歧视分数筛选)
│   │   └── FairCosG.py         # FairCosG/GuardFed ⭐
│   │
│   ├── models/                  # 模型和工具
│   │   └── function.py         # MLP、公平性指标
│   │
│   ├── data_loader.py          # 统一数据加载器 ⭐
│   └── HYPERPARAMETERS.py      # 全局配置
│
├── scripts/                     # 实验脚本
│   ├── run.py                  # 主训练脚本
│   └── test_setup.py           # 测试脚本
│
├── requirements.txt             # Python依赖
└── README.md                    # 本文件
```

---

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备数据

确保数据文件在正确位置：
```
data/adult/adult.data
data/adult/adult.test
data/compas/compas-scores-two-years.csv
```

### 3. 运行测试

```bash
python3 scripts/test_setup.py
```

### 4. 运行实验

```bash
python3 scripts/run.py
```

---

## 💻 核心功能

### 统一数据加载器 ⭐

**一行代码切换数据集：**

```python
from src.data_loader import DatasetLoader

# 加载Adult数据集
loader = DatasetLoader(dataset_name='adult', seed=123)

# 切换到COMPAS数据集
loader = DatasetLoader(dataset_name='compas', seed=123)

# 获取数据
tensors = loader.get_tensors()
test_loader = loader.create_test_loader()

# 分割服务器/客户端数据
server_df, client_df = loader.split_server_client_data(0.1)
client_data = loader.create_client_data_dict(client_df, num_clients=20)
```

### 算法实现

#### 1. FedAvg + Reweighting
- 基线联邦平均算法
- 支持样本重加权以提升公平性

#### 2. FLTrust
- 服务器维护IID数据集
- 通过余弦相似度评估客户端可信度

#### 3. FairFed
- 根据EOD动态调整客户端权重
- 公平性感知聚合

#### 4. Medium
- 中位数聚合
- 对异常值具有天然鲁棒性

#### 5. FairG
- 计算歧视分数
- K-Means聚类筛选恶意客户端

#### 6. FairCosG (GuardFed) ⭐
- 结合余弦相似度和公平性指标
- FairCos分数: `cos_sim * exp(-λ * EOD)`
- 本项目核心算法

### 攻击模型

**公平性攻击：**
- `attack_fair_1`: 属性翻转
- `attack_fair_2`: 混合翻转

**准确性攻击：**
- `attack_acc_0.5`: FOE攻击 (权重×-0.5)
- `attack_acc_LIE`: LIE攻击

**混合攻击：**
- `attack_super_mixed`: S-DFA (同时数据中毒+权重攻击)
- `mixed`: Sp-DFA (分工攻击)

---

## ⚙️ 配置说明

### 修改数据集

编辑 `scripts/run.py` 第37行：
```python
DATASET_NAME = 'compas'  # 改为COMPAS数据集
```

### 修改训练参数

编辑 `src/HYPERPARAMETERS.py`：
```python
HYPERPARAMETERS = {
    'NUM_GLOBAL_ROUNDS': 100,           # 训练轮数
    'NUM_CLIENTS': 20,                  # 客户端数量
    'BATCH_SIZE': 256,                  # 批量大小
    'FAIRCOSG_LAMBDA_VALUES': [20],     # FairCosG的λ值
    'ALPHA_DIRICHLET': [1.0],           # Non-IID程度
}
```

### 修改恶意客户端数量

编辑 `scripts/run.py` 第45行：
```python
MALICIOUS_CLIENT_NUMBERS = [2, 4, 6, 8, 10]
```

---

## 📊 公平性指标

### EOD (Equalized Odds Difference)
```
EOD = TPR_unprivileged - TPR_privileged
```
衡量不同群体的真正例率差异。

### SPD (Statistical Parity Difference)
```
SPD = P(pred=1 | unprivileged) - P(pred=1 | privileged)
```
衡量不同群体的正类预测率差异。

### Reweighing (RW)
```
W(s,c) = P(S=s) * P(Class=c) / P(S=s ∧ Class=c)
```
通过样本权重平衡训练数据偏见。

---

## 📈 实验结果

运行实验后会生成：
- `*_detailed_*.csv` - 每轮详细数据
- `*_summary_*.csv` - 最后5轮平均值
- `*_results_*.xlsx` - Excel报告
- `*_comparison_*.png` - 可视化图表

---

## ⏳ 当前状态

### 已完成 (70%)

✅ 项目结构重组  
✅ 统一数据加载器创建  
✅ 文档体系完善  
✅ 配置文件创建  

### 待完成 (30%)

⏳ 更新所有文件的导入语句 (2-3小时)  
⏳ 运行集成测试 (2-3小时)  
⏳ 验证结果一致性 (1-2小时)  

**详细步骤**: 查看项目根目录下的其他文档文件

---

## 🔧 开发指南

### 添加新数据集

在 `src/data_loader.py` 中添加新方法：

```python
def _load_newdataset(self):
    """加载新数据集"""
    # 1. 读取数据
    # 2. 数据预处理
    # 3. 设置敏感属性
    # 4. 返回处理后的数据
```

### 添加新算法

1. 在 `src/algorithms/` 创建新文件
2. 实现 `Client` 和 `Server` 类
3. 在 `scripts/run.py` 中导入
4. 在 `src/HYPERPARAMETERS.py` 添加配置

---

## 📝 依赖项

```
numpy>=1.21.0
pandas>=1.3.0
torch>=1.10.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
openpyxl>=3.0.0
tqdm>=4.62.0
```

---

## 🐛 常见问题

### Q: 为什么不能直接运行实验？
A: 需要先更新所有文件的导入语句，使其使用新的模块路径。

### Q: 如何切换数据集？
A: 在 `scripts/run.py` 中修改 `DATASET_NAME = 'compas'`。

### Q: 数据文件找不到？
A: 检查数据文件路径：
```bash
ls data/adult/adult.data
ls data/compas/compas-scores-two-years.csv
```

### Q: 如何添加新数据集？
A: 在 `DatasetLoader` 类中添加 `_load_newdataset()` 方法。

---

## 📚 项目统计

- **文档文件**: 1 个 (本文件)
- **Python源文件**: 13 个
- **总代码行数**: ~8,350 行
- **新增代码**: ~450 行 (data_loader.py)
- **算法数量**: 6 个
- **数据集**: 2 个

---

## 🎯 核心改进

### 1. 统一数据加载器 ⭐⭐⭐⭐⭐
从分散的数据加载逻辑到统一的 DatasetLoader 类，一行代码切换数据集。

### 2. 模块化结构 ⭐⭐⭐⭐⭐
从单层混乱结构到清晰的三层架构 (data/, src/, scripts/)。

### 3. 集中化配置 ⭐⭐⭐⭐⭐
从分散的配置到统一的 HYPERPARAMETERS.py。

### 4. 易于扩展 ⭐⭐⭐⭐⭐
添加新数据集/算法只需几步。

---

## 📞 联系方式

- 📧 邮箱: [您的邮箱]
- 🐛 问题反馈: [GitHub Issues]
- 💬 讨论: [GitHub Discussions]

---

## 📄 许可证

[您的许可证]

---

## 🙏 致谢

感谢所有为本项目做出贡献的研究者和开发者。

---

**GuardFed - 让联邦学习更公平、更安全！** 🛡️

---

**版本**: v2.0.0  
**最后更新**: 2026-02-02
