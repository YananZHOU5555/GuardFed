# GuardFed 表4 - 精确实现计划

## 📊 论文表4中的算法（COMPAS数据集）

根据论文表4，需要实现的算法如下：

### 1. Fairness (debias) algorithms - 公平性算法
- ✅ **FedAvg** - 已有结果（但可能不是RW版本）
- ✅ **FairFed** - 已有结果（但可能不是RW版本）

### 2. Robust FL algorithms for general adversarial attacks - 通用鲁棒算法
- ✅ **Median** - 已有结果
- ✅ **FLTrust** - 已有结果
- ❌ **Class-B FL** - 缺失

### 3. Robust FL for fairness attacks - 公平性攻击防御
- ❌ **FairGuard** - 缺失（但FairG.py可能就是）

### 4. Robust FL (hybrid) - 混合防御
- ❌ **FLTrust + FairGuard** - 缺失

### 5. Ours - 论文方法
- ❌ **GuardFed** - 缺失（最重要！）

---

## 📊 论文表4中的算法（ADULT数据集）

同样的算法列表，但数据集不同。

---

## 🔍 当前状态对比

### 已训练的算法（来自table4_final.json）
1. FedAvg_RW
2. FairFed_RW
3. Medium_RW
4. FLTrust_RW
5. FedAvg_RW_FairG
6. FLTrust_RW_FairG
7. FedAvg_RW_FairCosG

### 表4中需要的算法
1. FedAvg
2. FairFed
3. Median
4. FLTrust
5. Class-B FL
6. FairGuard
7. FLTrust + FairGuard (Hybrid)
8. GuardFed

---

## ⚠️ 关键问题

### 问题1: RW版本 vs 非RW版本
**已训练**: FedAvg_RW, FairFed_RW, Medium_RW, FLTrust_RW
**表4需要**: FedAvg, FairFed, Median, FLTrust

**可能性**:
1. 表4中的算法就是RW版本（只是名字简化了）
2. 表4中的算法是非RW版本（需要重新训练）

**验证方法**: 对比已训练结果和论文表4的数值

### 问题2: FairG vs FairGuard
**已有**: FairG.py, FedAvg_RW_FairG, FLTrust_RW_FairG
**表4需要**: FairGuard

**可能性**:
1. FairG = FairGuard（只是命名不同）
2. FairG是FairGuard的一部分

---

## 📋 精确实现计划

### 步骤1: 验证已有结果 ✓

对比 `table4_final.json` 和论文表4的数值：

#### ADULT IID Benign:
- **论文FedAvg**: ACC=83.05, AEOD=0.018, ASPD=0.104
- **已训练FedAvg_RW**: ACC=83.37, AEOD=0.048, ASPD=-0.103

**结论**: 数值不完全匹配，可能是：
1. 论文用的是非RW版本
2. 或者有其他参数差异

### 步骤2: 确定需要实现的算法

#### 必须实现（表4中有但缺失）⭐⭐⭐⭐⭐
1. **GuardFed** - 论文核心方法
2. **Class-B FL** - 表4中的对比方法
3. **FairGuard** - 如果FairG不是的话
4. **Hybrid (FLTrust + FairGuard)** - 混合防御

#### 可能需要实现（取决于验证结果）⭐⭐
5. **FedAvg (非RW)** - 如果表4用的是非RW版本
6. **FairFed (非RW)** - 如果表4用的是非RW版本
7. **Median (非RW)** - 如果表4用的是非RW版本
8. **FLTrust (非RW)** - 如果表4用的是非RW版本

---

## 🎯 优先级排序

### 第一优先级: GuardFed ⭐⭐⭐⭐⭐
**原因**: 论文的核心贡献，表4的主角
**状态**: 必须实现
**预计时间**: 2-3天

### 第二优先级: 验证FairG = FairGuard ⭐⭐⭐⭐
**原因**: 确定是否需要重新实现FairGuard
**状态**: 需要验证
**预计时间**: 0.5天

**验证方法**:
1. 检查FairG.py的实现
2. 对比FairGuard论文的算法描述
3. 运行FedAvg_RW_FairG，看结果是否接近表4的FairGuard

### 第三优先级: Class-B FL ⭐⭐⭐
**原因**: 表4中的对比方法
**状态**: 需要实现或查找
**预计时间**: 1-2天

**可能性**:
1. 可能是某个已知算法的别名
2. 需要查找原始论文

### 第四优先级: Hybrid ⭐⭐
**原因**: 证明简单组合不够
**状态**: 需要实现
**预计时间**: 0.5-1天

### 第五优先级: 非RW版本 ⭐
**原因**: 取决于验证结果
**状态**: 可能需要
**预计时间**: 0.5天

---

## 📝 立即行动清单

### Action 1: 对比数值验证 ✓
```python
# 对比已训练结果和论文表4
# 确定是否需要非RW版本
```

### Action 2: 检查FairG是否是FairGuard
```bash
# 查看FairG.py的实现
# 搜索FairGuard相关文献
```

### Action 3: 查找Class-B FL
```bash
# 搜索论文中Class-B FL的引用
# 确定是什么算法
```

### Action 4: 实现GuardFed
```python
# 这是最重要的，必须实现
```

---

## 🤔 需要用户确认的问题

1. **表4中的FedAvg/FairFed等是RW版本还是非RW版本？**
   - 如果是RW版本，我们已经有了
   - 如果是非RW版本，需要重新训练

2. **FairG是否就是FairGuard？**
   - 需要验证FairG.py的实现
   - 或者用户确认

3. **Class-B FL是什么算法？**
   - 需要查找论文引用
   - 或者用户提供信息

4. **是否需要完全匹配论文表4的数值？**
   - 还是只需要趋势一致即可

---

## 建议的下一步

**选项A: 先验证再实现**
1. 对比数值，确定RW vs 非RW
2. 验证FairG = FairGuard
3. 查找Class-B FL
4. 然后实现GuardFed

**选项B: 直接实现GuardFed**
1. 先实现最重要的GuardFed
2. 然后再处理其他算法
3. 最后验证和调整

**我的建议**: 选项B
- GuardFed是最重要的
- 其他算法可以边实现边验证
- 不要被细节阻塞主要目标

---

## 用户需要回答的问题

1. **表4中的算法是否都需要RW？**
2. **FairG是否就是表4中的FairGuard？**
3. **Class-B FL是什么？需要实现吗？**
4. **是否直接开始实现GuardFed？**
