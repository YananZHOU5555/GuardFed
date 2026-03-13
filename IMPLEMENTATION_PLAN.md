# GuardFed 表4完整复现 - 实现计划

## 当前状态分析

### ✅ 已实现并训练完成的算法 (7个)
根据 `results/table4_final.json`，以下算法已完成：

1. **FedAvg_RW** - FedAvg with Reweighting
2. **FairFed_RW** - FairFed with Reweighting
3. **Medium_RW** - Median aggregation with Reweighting
4. **FLTrust_RW** - FLTrust with Reweighting
5. **FedAvg_RW_FairG** - FedAvg + Reweighting + FairG
6. **FLTrust_RW_FairG** - FLTrust + Reweighting + FairG
7. **FedAvg_RW_FairCosG** - FedAvg + Reweighting + FairCosG

### ❌ 缺失的算法 (根据论文表4)

从论文表4可以看出，需要对比的算法包括：

#### 基础算法组
1. **FedAvg** (无RW) - 基础联邦平均
2. **FairFed** (无RW) - 公平性感知联邦学习
3. **Medium** (无RW) - 中位数聚合
4. **FLTrust** (无RW) - 基于信任的联邦学习

#### 防御算法组
5. **FairGuard** - 论文中的主要对比方法（防御公平性攻击）
6. **Hybrid** - 混合防御（结合性能和公平性防御）
7. **GuardFed** - 论文提出的新方法（主角）

---

## 实现计划

### 阶段1: 补充基础算法（无RW版本）✓

**优先级**: 中
**原因**: 论文表4中包含了无RW版本的对比

#### 1.1 FedAvg (无RW)
- **文件**: 已存在 `src/algorithms/FedAvg.py`
- **状态**: 需要确认是否支持无RW模式
- **实现**: 修改Client类，添加参数控制是否使用RW

#### 1.2 FairFed (无RW)
- **文件**: 已存在 `src/algorithms/FairFed.py`
- **状态**: 需要确认是否支持无RW模式
- **实现**: 修改Client类，添加参数控制是否使用RW

#### 1.3 Medium (无RW)
- **文件**: 已存在 `src/algorithms/Medium.py`
- **状态**: 需要确认是否支持无RW模式
- **实现**: 修改Client类，添加参数控制是否使用RW

#### 1.4 FLTrust (无RW)
- **文件**: 已存在 `src/algorithms/FLTrust.py`
- **状态**: 需要确认是否支持无RW模式
- **实现**: 修改Client类，添加参数控制是否使用RW

---

### 阶段2: 实现FairGuard ⭐⭐⭐

**优先级**: 高
**原因**: 论文中的主要对比方法，必须实现

#### 2.1 FairGuard算法理解
- **论文引用**: FairGuard (Sheng et al., 2024)
- **核心思想**: 使用聚类方法检测和过滤恶意客户端
- **需要**:
  - 随机生成数据集进行评估
  - 基于聚类结果筛选客户端
  - 恢复模型公平性

#### 2.2 实现步骤
1. 创建 `src/algorithms/FairGuard.py`
2. 实现FairGuard的Client类
3. 实现FairGuard的Server类
4. 实现聚类筛选逻辑
5. 测试验证

#### 2.3 参考资料
- 检查是否有FairGuard的原始代码
- 阅读FairGuard论文了解算法细节
- 可能需要实现K-means聚类

---

### 阶段3: 实现Hybrid防御 ⭐⭐

**优先级**: 中高
**原因**: 论文中用于证明简单组合不够有效

#### 3.1 Hybrid算法理解
- **定义**: 结合性能防御（如FLTrust）和公平性防御（如FairGuard）
- **实现方式**:
  - 可能是FLTrust + FairGuard的组合
  - 或者是FLTrust + FairFed的组合

#### 3.2 实现步骤
1. 确定Hybrid的具体组合方式
2. 创建 `src/algorithms/Hybrid.py`
3. 整合两种防御机制
4. 测试验证

---

### 阶段4: 实现GuardFed（论文核心） ⭐⭐⭐⭐⭐

**优先级**: 最高
**原因**: 论文提出的新方法，核心贡献

#### 4.1 GuardFed算法理解
从论文中提取的关键信息：

**核心机制**:
1. **双视角信任评分**:
   - 效用偏差 (Utility Deviation)
   - 公平性退化 (Fairness Degradation)

2. **公平性感知参考模型**:
   - 使用少量干净服务器数据
   - 可以用生成AI增强数据

3. **自适应聚合**:
   - 计算每个客户端的信任分数
   - 只聚合超过阈值的更新

**信任分数公式** (从论文中):
```
Trust_i = ReLU(cos(g_i, g_s)) * exp(-τ * Fair_i)
```
其中:
- `g_i`: 客户端i的更新
- `g_s`: 服务器参考更新
- `Fair_i`: 客户端i的公平性偏差
- `τ`: 平衡系数

#### 4.2 实现步骤
1. 创建 `src/algorithms/GuardFed.py`
2. 实现GuardFed的Client类
3. 实现GuardFed的Server类，包括:
   - 服务器参考模型训练
   - 双视角信任评分计算
   - 自适应聚合机制
4. 实现公平性评估函数
5. 测试验证

#### 4.3 关键参数
- `τ` (tau): 平衡系数，默认值需要从论文中确定
- `Γ` (Gamma): 信任阈值
- 服务器数据比例: 1%-10%

---

## 实现优先级排序

### 第一优先级: GuardFed ⭐⭐⭐⭐⭐
**原因**: 论文核心贡献，必须实现
**预计工作量**: 2-3天
**依赖**: 需要理解论文算法细节

### 第二优先级: FairGuard ⭐⭐⭐
**原因**: 主要对比方法
**预计工作量**: 1-2天
**依赖**: 可能需要查找原始论文/代码

### 第三优先级: Hybrid ⭐⭐
**原因**: 证明简单组合不够
**预计工作量**: 0.5-1天
**依赖**: 需要FairGuard实现完成

### 第四优先级: 无RW版本 ⭐
**原因**: 补充对比实验
**预计工作量**: 0.5天
**依赖**: 修改现有代码即可

---

## 检查清单

### 步骤1: 检查现有代码 ✓
- [x] 查看已训练结果
- [ ] 检查FairGuard是否已有实现
- [ ] 检查GuardFed是否已有实现
- [ ] 检查Hybrid是否已有实现
- [ ] 确认无RW版本的支持情况

### 步骤2: 查找参考资料
- [ ] 查找FairGuard原始论文
- [ ] 查找FairGuard原始代码（如果有）
- [ ] 仔细阅读GuardFed论文的算法部分
- [ ] 提取GuardFed的伪代码和公式

### 步骤3: 实现算法
- [ ] 实现GuardFed
- [ ] 实现FairGuard
- [ ] 实现Hybrid
- [ ] 添加无RW版本支持

### 步骤4: 测试验证
- [ ] 快速测试（1轮）
- [ ] 完整测试（50轮）
- [ ] 对比论文结果

### 步骤5: 生成表4
- [ ] 运行所有算法组合
- [ ] 生成结果表格
- [ ] 对比论文表4

---

## 当前行动建议

### 立即行动:
1. **检查现有代码中是否有FairGuard/GuardFed的实现**
   ```bash
   ls src/algorithms/
   grep -r "FairGuard\|GuardFed" src/
   ```

2. **阅读论文中GuardFed的算法描述**
   - 提取伪代码
   - 理解信任评分公式
   - 确定超参数

3. **制定详细的实现计划**
   - 先实现GuardFed（最重要）
   - 再实现FairGuard（主要对比）
   - 最后补充其他算法

---

## 预期结果

完成后，表4应该包含以下算法的结果：

| 算法类别 | 算法名称 | 状态 |
|---------|---------|------|
| 基础算法 | FedAvg | ❌ 待实现 |
| 基础算法 | FedAvg_RW | ✅ 已完成 |
| 基础算法 | FairFed | ❌ 待实现 |
| 基础算法 | FairFed_RW | ✅ 已完成 |
| 基础算法 | Medium | ❌ 待实现 |
| 基础算法 | Medium_RW | ✅ 已完成 |
| 鲁棒算法 | FLTrust | ❌ 待实现 |
| 鲁棒算法 | FLTrust_RW | ✅ 已完成 |
| 防御算法 | FairGuard | ❌ 待实现 |
| 防御算法 | FedAvg_RW_FairG | ✅ 已完成 |
| 防御算法 | FLTrust_RW_FairG | ✅ 已完成 |
| 防御算法 | FedAvg_RW_FairCosG | ✅ 已完成 |
| 混合防御 | Hybrid | ❌ 待实现 |
| 论文方法 | GuardFed | ❌ 待实现 |

**总计**: 7/14 完成 (50%)

---

## 下一步

请确认：
1. 是否先检查现有代码中有没有FairGuard/GuardFed的实现？
2. 是否需要我先实现GuardFed（最重要的算法）？
3. 还是先补充其他缺失的算法？
