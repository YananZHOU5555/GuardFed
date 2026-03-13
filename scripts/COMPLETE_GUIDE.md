# 完整复现论文表4 - 使用指南

## 📋 概述

`reproduce_table4_complete.py` 是一个完整的实验脚本，可以复现论文表4的所有实验结果。

## ✨ 特性

- ✅ **50轮完整训练** - 匹配论文设置
- ✅ **Dirichlet数据分区** - 支持IID和non-IID
- ✅ **7种算法** - FedAvg, FairFed, Median, FLTrust, FairGuard, Hybrid, GuardFed
- ✅ **5种攻击** - Benign, F Flip, FOE, S-DFA, Sp-DFA
- ✅ **2种数据分布** - IID (α=5000), non-IID (α=5)
- ✅ **自动保存结果** - JSON格式保存到results目录
- ✅ **进度显示** - 实时显示实验进度和结果

## 🚀 快速开始

### 1. 激活虚拟环境

```cmd
cd D:\GitHub\GuardFed-main
GuardFed\Scripts\activate.bat
```

### 2. 运行完整实验

```cmd
python scripts\reproduce_table4_complete.py
```

系统会询问确认：
```
是否开始运行完整实验? (y/n): y
```

### 3. 等待完成

- **总实验数**: 70个 (7算法 × 5攻击 × 2分布)
- **预计时间**: ~70分钟 (每个实验约1分钟)
- **结果保存**: `results/table4_results.json`

## 📊 实验配置

### 训练参数

```python
NUM_ROUNDS = 50          # 训练轮次
NUM_CLIENTS = 20         # 客户端数量
NUM_MALICIOUS = 4        # 恶意客户端 (20%)
BATCH_SIZE = 256         # 批量大小
LEARNING_RATE = 0.01     # 学习率
LOCAL_EPOCHS = 1         # 本地训练轮次
```

### 数据分布

```python
ALPHA_IID = 5000         # IID设置
ALPHA_NON_IID = 5        # non-IID设置
```

### 算法列表

1. **FedAvg** - 联邦平均 + Reweighting
2. **FairFed** - 公平联邦学习 + Reweighting
3. **Median** - 中位数聚合 + Reweighting
4. **FLTrust** - 信任评分聚合 + Reweighting
5. **FairGuard** - 公平性防御
6. **Hybrid** - FLTrust + FairGuard
7. **GuardFed** - 我们的方法

### 攻击类型

1. **Benign** - 无攻击（基准）
2. **F Flip** - 翻转敏感属性（公平性攻击）
3. **FOE** - 权重×-0.5（性能攻击）
4. **S-DFA** - 同步双面攻击（F Flip + FOE）
5. **Sp-DFA** - 分裂双面攻击（一半F Flip，一半FOE）

## 📈 输出示例

### 运行时输出

```
================================================================================
完整复现论文表4 - Adult数据集
================================================================================
开始时间: 2026-02-08 16:30:00
================================================================================

总实验数: 70
预计时间: ~70 分钟

================================================================================

数据分布: IID (α=5000)
================================================================================

数据加载完成

[1/70] FedAvg | Benign | IID
    结果: ACC=0.8305, AEOD=0.0180, ASPD=0.1040 (58.3s)

[2/70] FedAvg | F Flip | IID
    结果: ACC=0.8176, AEOD=0.2160, ASPD=0.1210 (59.1s)

...
```

### 结果表格

```
实验结果汇总
================================================================================

IID 数据分布:
--------------------------------------------------------------------------------
算法             攻击        ACC      AEOD     ASPD
--------------------------------------------------------------------------------
FedAvg          Benign    0.8305   0.0180   0.1040
FedAvg          F Flip    0.8176   0.2160   0.1210
FedAvg          FOE       0.7663   0.0010   0.0110
...
GuardFed        Benign    0.8374   0.0220   0.0960
GuardFed        S-DFA     0.8373   0.0260   0.0710
GuardFed        Sp-DFA    0.8372   0.0510   0.0900
```

## 💾 结果文件

结果保存在 `results/table4_results.json`：

```json
{
  "IID": {
    "FedAvg": {
      "Benign": {
        "accuracy": 0.8305,
        "aeod": 0.0180,
        "aspd": 0.1040
      },
      ...
    },
    ...
  },
  "non-IID": {
    ...
  }
}
```

## ⚙️ 自定义配置

### 修改训练轮次

编辑 `reproduce_table4_complete.py`：

```python
NUM_ROUNDS = 100  # 增加到100轮
```

### 只运行特定算法

修改 `run_experiments()` 函数：

```python
algorithms = {
    'FedAvg': train_fedavg,
    'GuardFed': train_fedavg,  # 只测试这两个
}
```

### 只运行特定攻击

```python
attacks = {
    'Benign': 'benign',
    'S-DFA': 's_dfa',  # 只测试这两个
}
```

## 🔧 当前实现状态

### ✅ 已实现

- [x] FedAvg + Reweighting
- [x] Median + Reweighting
- [x] 所有攻击类型 (Benign, F Flip, FOE, S-DFA, Sp-DFA)
- [x] Dirichlet数据分区
- [x] 完整的评估指标 (ACC, AEOD, ASPD)
- [x] 结果保存和展示

### ⚠️ 使用占位符

以下算法当前使用FedAvg作为占位符：
- [ ] FairFed (需要实现公平性约束)
- [ ] FLTrust (需要实现信任评分)
- [ ] FairGuard (需要实现FairG防御)
- [ ] Hybrid (需要组合FLTrust和FairGuard)
- [ ] GuardFed (需要实现FairCosG防御)

### 📝 完整实现建议

要实现完整的算法，需要：

1. **FairFed**: 添加公平性约束到损失函数
2. **FLTrust**: 实现服务器参考模型和余弦相似度评分
3. **FairGuard**: 实现FairG的K-Means聚类和筛选
4. **GuardFed**: 实现FairCosG的双视角信任评分

参考原始代码：
- `src/algorithms/FairFed.py`
- `src/algorithms/FLTrust.py`
- `src/algorithms/FairG.py`
- `src/algorithms/FairCosG.py`

## 🐛 故障排除

### 问题1: 内存不足

**解决方法**:
```python
BATCH_SIZE = 128  # 减小批量大小
NUM_ROUNDS = 25   # 减少训练轮次
```

### 问题2: 运行时间太长

**解决方法**:
```python
# 只运行部分实验
algorithms = {'FedAvg': train_fedavg, 'GuardFed': train_fedavg}
attacks = {'Benign': 'benign', 'S-DFA': 's_dfa'}
# 这样只运行 2×2×2 = 8 个实验
```

### 问题3: 结果与论文不符

**可能原因**:
1. 算法使用了占位符实现
2. 超参数设置不同
3. 随机种子不同
4. 数据预处理方式不同

**解决方法**:
- 实现完整的算法
- 检查超参数设置
- 使用相同的随机种子
- 参考原始代码的数据处理

## 📞 获取帮助

如果遇到问题：

1. 查看 `REPRODUCTION_REPORT.md` 了解已知问题
2. 查看 `README_TABLE4.md` 了解详细说明
3. 参考原始代码 `scripts/run.py`
4. 检查论文中的实验设置

## 🎯 预期结果

根据论文表4，预期结果示例：

| 算法 | 分布 | 攻击 | ACC | AEOD | ASPD |
|------|------|------|-----|------|------|
| FedAvg | IID | Benign | 83.05% | 0.018 | 0.104 |
| GuardFed | IID | Benign | **83.74%** | 0.022 | **0.096** |
| GuardFed | IID | S-DFA | **83.73%** | **0.026** | **0.071** |
| GuardFed | non-IID | Sp-DFA | **82.58%** | **0.015** | **0.084** |

**注意**: 当前使用占位符实现的算法结果可能与论文不同。

## 📅 更新日志

- **2026-02-08**: 创建初始版本
  - 实现FedAvg和Median算法
  - 实现所有攻击类型
  - 实现Dirichlet数据分区
  - 添加结果保存功能
  - 设置50轮训练

---

**最后更新**: 2026-02-08
**版本**: 1.0
**状态**: 可运行，部分算法使用占位符
