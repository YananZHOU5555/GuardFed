# 复现论文表4实验指南

## 概述

本目录包含用于复现论文表4（Adult数据集）实验结果的脚本。

## 论文表4内容

表4展示了在Adult数据集上，不同鲁棒联邦学习方法在各种攻击场景下的性能对比：

- **数据分布**: IID (α=5000) 和 non-IID (α=5)
- **攻击类型**: Benign, F Flip, FOE, S-DFA, Sp-DFA
- **评估指标**: 准确率(ACC), AEOD, ASPD
- **对比方法**: FedAvg, FairFed, Median, FLTrust, FairGuard, FLTrust+FairGuard, GuardFed

## 文件说明

### 1. `reproduce_table4_full.py` (推荐)

**用途**: 快速验证实验流程

**特点**:
- 交互式脚本，会询问是否继续
- 只运行一个简单的测试用例 (FedAvg + Benign + IID)
- 训练轮次少 (5轮)，快速验证
- 提供详细的实验说明和下一步建议

**运行方法**:
```bash
cd D:\GitHub\GuardFed-main
GuardFed\Scripts\activate.bat
python scripts\reproduce_table4_full.py
```

**预期输出**:
```
复现论文表4 - Adult数据集完整实验
================================================================================
开始时间: 2026-02-08 XX:XX:XX
================================================================================

实验配置:
- 数据集: Adult Income
- 客户端数量: 20
- 恶意客户端: 4 (20%)
...

是否继续运行快速验证测试? (y/n): y

开始快速验证测试...

================================================================================
快速验证: FedAvg + Benign + IID
================================================================================

1. 加载Adult数据集...
   ✓ 数据加载成功: 30162 训练样本, 15059 测试样本

2. 数据分区...
   ✓ 服务器数据: 1508 样本
   ✓ 每个客户端: ~1432 样本

3. 初始化全局模型...
   ✓ 模型参数: 258

4. 开始训练 (5 轮)...
   轮次 2/5: ACC=0.7xxx, EOD=0.xxx, SPD=0.xxx
   轮次 4/5: ACC=0.7xxx, EOD=0.xxx, SPD=0.xxx

5. 最终评估...

最终结果:
  准确率 (ACC): 0.xxxx
  公平性 (AEOD): 0.xxxx
  公平性 (ASPD): 0.xxxx
```

### 2. `reproduce_table4.py`

**用途**: 实验框架代码

**特点**:
- 包含完整的实验框架
- 定义了所有算法和攻击类型
- 提供了数据加载和分区的完整实现
- 目前只实现了FedAvg的简化版本

**使用方法**:
```bash
python scripts\reproduce_table4.py
```

## 完整实验步骤

要获得与论文完全一致的结果，需要以下步骤：

### 步骤1: 验证环境

```bash
cd D:\GitHub\GuardFed-main
GuardFed\Scripts\activate.bat
python scripts\verify_setup.py
```

确保所有依赖都已正确安装。

### 步骤2: 快速验证

```bash
python scripts\reproduce_table4_full.py
```

运行快速验证，确保实验流程可行。

### 步骤3: 实现完整算法

需要实现以下算法的完整版本：

1. **FedAvg + Reweighting** ✓ (已有简化实现)
2. **FairFed + Reweighting** - 需要实现
3. **Median + Reweighting** - 需要实现
4. **FLTrust + Reweighting** - 需要实现
5. **FairGuard** - 需要实现
6. **FLTrust + FairGuard** - 需要实现
7. **GuardFed** - 需要实现

参考原始代码:
- `src/algorithms/FedAvg.py`
- `src/algorithms/FairFed.py`
- `src/algorithms/Medium.py`
- `src/algorithms/FLTrust.py`
- `src/algorithms/FairG.py`
- `src/algorithms/FairCosG.py`

### 步骤4: 实现攻击类型

需要实现以下攻击：

1. **Benign** (无攻击) ✓
2. **F Flip** (翻转敏感属性) - 部分实现
3. **FOE** (权重×-0.5) - 部分实现
4. **S-DFA** (F Flip + FOE) - 需要实现
5. **Sp-DFA** (一半F Flip, 一半FOE) - 需要实现

### 步骤5: 运行完整实验

修改配置参数：
```python
NUM_ROUNDS = 50  # 或 100，根据论文设置
```

运行完整实验矩阵：
- 7个算法
- 5种攻击
- 2种数据分布
- 总共 70 个实验

### 步骤6: 结果对比

将实验结果与论文表4对比：

**论文中的期望结果 (部分示例)**:

| 方法 | 分布 | 攻击 | ACC | AEOD | ASPD |
|------|------|------|-----|------|------|
| FedAvg | IID | Benign | 83.05 | 0.018 | 0.104 |
| FedAvg | IID | F Flip | 81.76 | 0.216 | 0.121 |
| GuardFed | IID | Benign | **83.74** | 0.022 | 0.096 |
| GuardFed | IID | S-DFA | **83.73** | **0.026** | **0.071** |
| GuardFed | non-IID | Sp-DFA | **82.58** | **0.015** | **0.084** |

## 常见问题

### Q1: 为什么结果与论文不一致？

**可能原因**:
1. 训练轮次不足（快速测试只用5轮，论文用50-100轮）
2. 使用了简化的算法实现
3. 超参数设置不同
4. 随机种子不同

**解决方法**:
- 增加训练轮次
- 使用原始算法实现
- 检查超参数设置
- 使用相同的随机种子

### Q2: 如何加速实验？

**建议**:
1. 减少训练轮次（用于快速验证）
2. 减少客户端数量
3. 使用GPU加速（如果可用）
4. 并行运行多个实验

### Q3: 内存不足怎么办？

**解决方法**:
1. 减小批量大小
2. 减少客户端数量
3. 使用梯度累积
4. 分批运行实验

## 实验配置参数

### 关键参数

```python
# 数据集
DATASET_NAME = 'adult'

# 联邦学习设置
NUM_CLIENTS = 20          # 客户端数量
NUM_MALICIOUS = 4         # 恶意客户端数量 (20%)
NUM_ROUNDS = 50           # 训练轮次（完整实验）

# 训练参数
BATCH_SIZE = 256          # 批量大小
LEARNING_RATE = 0.01      # 学习率
SERVER_DATA_RATIO = 0.05  # 服务器数据比例 (5%)

# 数据分布
ALPHA_IID = 5000          # IID设置
ALPHA_NON_IID = 5         # non-IID设置
```

### 攻击参数

```python
# FOE攻击
FOE_SCALE = -0.5          # 权重缩放因子

# S-DFA攻击
# 所有恶意客户端执行: F Flip + FOE

# Sp-DFA攻击
# 前一半恶意客户端: F Flip
# 后一半恶意客户端: FOE
```

## 预期运行时间

**快速验证** (5轮):
- 单个实验: ~2-5分钟
- 总时间: ~5分钟

**完整实验** (50轮):
- 单个实验: ~20-50分钟
- 总时间: ~24-60小时 (70个实验)

**建议**: 先运行快速验证，确认流程正确后再运行完整实验。

## 输出格式

实验结果将以以下格式输出：

```
结果摘要:
--------------------------------------------------------------------------------

IID:
  FedAvg:
    Benign: ACC=0.8305, AEOD=0.0180, ASPD=0.1040
    F Flip: ACC=0.8176, AEOD=0.2160, ASPD=0.1210
    ...
  GuardFed:
    Benign: ACC=0.8374, AEOD=0.0220, ASPD=0.0960
    S-DFA: ACC=0.8373, AEOD=0.0260, ASPD=0.0710
    ...

non-IID:
  ...
```

## 联系与支持

如有问题，请参考：
- 论文: GuardFed: A Trustworthy Federated Learning Framework Against Dual-Facet Attacks
- 代码仓库: https://github.com/YananZHOU5555/GuardFed
- 原始实验脚本: `scripts/run.py`

## 更新日志

- 2026-02-08: 创建初始版本
  - 添加快速验证脚本
  - 添加实验框架代码
  - 添加使用说明文档
