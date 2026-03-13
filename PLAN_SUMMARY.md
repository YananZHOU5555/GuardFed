# GuardFed 表4完整复现 - 实施计划总结

## 📊 当前状态

### ✅ 已完成的算法 (7个)
1. FedAvg_RW
2. FairFed_RW
3. Medium_RW
4. FLTrust_RW
5. FedAvg_RW_FairG (FedAvg + Reweighting + FairGuard)
6. FLTrust_RW_FairG (FLTrust + Reweighting + FairGuard)
7. FedAvg_RW_FairCosG (FedAvg + Reweighting + FairCosG)

### 🔍 重要发现
- **FairG.py** = FairGuard的实现（使用聚类检测恶意客户端）
- **FairCosG.py** = FairGuard的余弦相似度变体

### ❌ 缺失的算法

#### 1. 无RW版本 (4个)
- FedAvg (无Reweighting)
- FairFed (无Reweighting)
- Medium (无Reweighting)
- FLTrust (无Reweighting)

#### 2. Hybrid防御 (1个)
- Hybrid (结合性能和公平性防御)

#### 3. GuardFed (1个) ⭐⭐⭐⭐⭐
- **论文核心贡献**，必须实现

---

## 🎯 实施计划

### 阶段1: 实现GuardFed（最高优先级）⭐⭐⭐⭐⭐

**为什么最重要**:
- 论文的核心贡献
- 表4的主角
- 需要与所有其他方法对比

**算法核心**:
```python
# 双视角信任评分
Trust_i = ReLU(cos(g_i, g_s)) * exp(-τ * Fair_i)

# 其中:
# - g_i: 客户端i的梯度更新
# - g_s: 服务器参考模型的梯度
# - Fair_i: 客户端i在服务器数据上的公平性偏差
# - τ: 平衡系数（控制公平性权重）
```

**实现步骤**:
1. 创建 `src/algorithms/GuardFed.py`
2. 实现GuardFed Client类（继承现有Client）
3. 实现GuardFed Server类，包括:
   - 服务器参考模型训练
   - 双视角信任评分计算
   - 自适应聚合（只聚合高信任分数的客户端）
4. 实现公平性评估函数
5. 测试验证

**关键参数**:
- `τ` (tau): 平衡系数，论文中测试了0.1到50
- `Γ` (Gamma): 信任阈值
- 服务器数据比例: 1%-10%（论文中测试了多个值）

**预计工作量**: 2-3天

---

### 阶段2: 实现Hybrid防御 ⭐⭐

**定义**: 结合FLTrust（性能防御）和FairGuard（公平性防御）

**实现方式**:
```python
# 可能的实现方式1: FLTrust + FairGuard串联
# 1. 先用FLTrust计算信任分数
# 2. 再用FairGuard进行公平性筛选
# 3. 聚合通过两个筛选的客户端

# 可能的实现方式2: FLTrust + FairFed
# 1. 使用FLTrust的聚合方式
# 2. 使用FairFed的公平性权重
```

**实现步骤**:
1. 确定Hybrid的具体定义（查看论文）
2. 创建 `src/algorithms/Hybrid.py`
3. 整合两种防御机制
4. 测试验证

**预计工作量**: 0.5-1天

---

### 阶段3: 添加无RW版本支持 ⭐

**目的**: 补充对比实验，展示Reweighting的作用

**实现方式**:
- 修改现有Client类，添加 `use_reweighting` 参数
- 当 `use_reweighting=False` 时，不使用样本权重

**实现步骤**:
1. 修改 `FedAvg.py` 的Client类
2. 修改 `FairFed.py` 的Client类
3. 修改 `Medium.py` 的Client类
4. 修改 `FLTrust.py` 的Client类
5. 更新实验脚本，添加无RW版本

**预计工作量**: 0.5天

---

## 📋 详细任务清单

### 任务1: 实现GuardFed ⭐⭐⭐⭐⭐
- [ ] 1.1 阅读论文中GuardFed的算法描述（Section 5）
- [ ] 1.2 提取伪代码和公式
- [ ] 1.3 创建 `src/algorithms/GuardFed.py`
- [ ] 1.4 实现GuardFed Client类
- [ ] 1.5 实现GuardFed Server类
  - [ ] 服务器参考模型训练
  - [ ] 余弦相似度计算
  - [ ] 公平性偏差计算
  - [ ] 信任评分计算
  - [ ] 自适应聚合
- [ ] 1.6 快速测试（1轮，1个攻击）
- [ ] 1.7 完整测试（50轮，5个攻击）

### 任务2: 实现Hybrid防御 ⭐⭐
- [ ] 2.1 确定Hybrid的具体定义
- [ ] 2.2 创建 `src/algorithms/Hybrid.py`
- [ ] 2.3 整合FLTrust和FairGuard
- [ ] 2.4 测试验证

### 任务3: 添加无RW版本 ⭐
- [ ] 3.1 修改FedAvg Client类
- [ ] 3.2 修改FairFed Client类
- [ ] 3.3 修改Medium Client类
- [ ] 3.4 修改FLTrust Client类
- [ ] 3.5 更新实验脚本
- [ ] 3.6 测试验证

### 任务4: 运行完整实验
- [ ] 4.1 更新 `run_table4_final.py`，添加新算法
- [ ] 4.2 运行所有算法组合（14算法 × 5攻击 × 2分布 = 140实验）
- [ ] 4.3 生成结果JSON
- [ ] 4.4 生成表4格式的输出

### 任务5: 结果验证
- [ ] 5.1 对比论文表4的数值
- [ ] 5.2 检查趋势是否一致
- [ ] 5.3 分析差异原因

---

## 🚀 立即行动

### 第一步: 理解GuardFed算法
```bash
# 阅读论文Section 5
# 提取关键公式和伪代码
# 理解双视角信任评分的计算方式
```

### 第二步: 检查论文中的算法描述
需要从论文中提取：
1. GuardFed的完整算法流程
2. 信任评分的计算公式
3. 服务器参考模型的训练方式
4. 公平性评估的具体方法
5. 超参数的默认值

### 第三步: 开始实现
从GuardFed开始，因为它是最重要的算法。

---

## 📈 预期完成时间

| 任务 | 预计时间 | 优先级 |
|------|---------|--------|
| GuardFed实现 | 2-3天 | ⭐⭐⭐⭐⭐ |
| Hybrid实现 | 0.5-1天 | ⭐⭐ |
| 无RW版本 | 0.5天 | ⭐ |
| 完整实验 | 2-3小时 | - |
| **总计** | **3-5天** | - |

---

## ❓ 需要确认的问题

1. **是否立即开始实现GuardFed？**
   - 这是最重要的算法
   - 需要仔细阅读论文理解算法细节

2. **Hybrid的具体定义是什么？**
   - 需要查看论文中的描述
   - 可能是FLTrust + FairGuard的组合

3. **无RW版本的优先级？**
   - 可以最后实现
   - 主要用于对比实验

---

## 下一步建议

**建议顺序**:
1. ✅ 先检查现有代码（已完成）
2. 📖 仔细阅读论文中GuardFed的算法描述
3. 💻 实现GuardFed
4. 🧪 测试GuardFed
5. 💻 实现Hybrid
6. 💻 添加无RW版本
7. 🚀 运行完整实验
8. 📊 生成表4

**是否开始实现GuardFed？**
