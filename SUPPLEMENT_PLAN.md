# GuardFed 表4补充实现计划

## 📊 当前状态

### ✅ 已完成（7个算法）
1. FedAvg_RW
2. FairFed_RW
3. Medium_RW
4. FLTrust_RW
5. FedAvg_RW_FairG
6. FLTrust_RW_FairG
7. FedAvg_RW_FairCosG

### ❌ 需要补充（根据表4）

#### 1. 非RW版本（4个）
- FedAvg (无Reweighting)
- FairFed (无Reweighting)
- Median (无Reweighting)
- FLTrust (无Reweighting)

#### 2. FairGuard独立版本（1个）
- FairGuard (如果FairG不等同于FairGuard)

#### 3. Hybrid混合防御（1个）
- FLTrust + FairGuard

#### 4. GuardFed（1个）⭐⭐⭐⭐⭐
- 论文核心方法

**总计**: 需要补充 6-7 个算法

---

## 🎯 实施计划

### 阶段1: 添加非RW版本支持 ⭐⭐

**目标**: 让现有算法支持无Reweighting模式

**实现方式**:
```python
# 在Client类初始化时添加参数
class Client:
    def __init__(self, ..., use_reweighting=True):
        self.use_reweighting = use_reweighting

    def local_train_fedavg(self, ...):
        if self.use_reweighting and self.sample_weights is not None:
            # 使用样本权重
            loss = loss * sample_weights_batch
        else:
            # 不使用样本权重
            loss = loss.mean()
```

**需要修改的文件**:
1. `src/algorithms/FedAvg.py` - Client类
2. `src/algorithms/FairFed.py` - Client类
3. `src/algorithms/Medium.py` - Client类
4. `src/algorithms/FLTrust.py` - Client类

**预计时间**: 0.5天

---

### 阶段2: 验证和实现FairGuard ⭐⭐⭐

**步骤2.1: 验证FairG是否是FairGuard**

检查点：
1. FairG.py的实现逻辑
2. 是否使用聚类检测恶意客户端
3. 是否使用随机生成数据评估

**步骤2.2: 如果FairG = FairGuard**
- 创建FairGuard算法（使用FairG）
- 测试验证

**步骤2.3: 如果FairG ≠ FairGuard**
- 查找FairGuard原始论文
- 实现FairGuard算法
- 测试验证

**预计时间**: 0.5-1天

---

### 阶段3: 实现Hybrid防御 ⭐⭐

**定义**: FLTrust + FairGuard的组合

**实现方式**:
```python
# src/algorithms/Hybrid.py

class HybridServer:
    def __init__(self, global_model, clients, algorithm, hyperparams,
                 server_data=None, fairg=None):
        # 初始化FLTrust组件
        self.fltrust_server = FLTrustServer(...)

        # 初始化FairGuard组件
        self.fairg = fairg or FairG(R=500)

    def run_round(self, round_num, ...):
        # 1. 客户端本地训练
        client_updates = {}
        for client in self.clients:
            update = client.local_train_fltrust(...)
            client_updates[client.client_id] = update

        # 2. FLTrust计算信任分数
        trust_scores = self.compute_fltrust_scores(client_updates)

        # 3. FairGuard进行公平性筛选
        fairness_scores = self.compute_fairness_scores(client_updates)
        selected_clients = self.fairg.filter_clients(fairness_scores)

        # 4. 结合两个分数进行聚合
        final_selected = [c for c in selected_clients
                         if trust_scores[c] > threshold]

        # 5. 聚合
        self.aggregate(client_updates, final_selected)
```

**预计时间**: 1天

---

### 阶段4: 实现GuardFed ⭐⭐⭐⭐⭐

**这是最重要的算法！**

#### 4.1 算法理解

从论文中提取的核心机制：

**双视角信任评分**:
```python
Trust_i = ReLU(cos(g_i, g_s)) * exp(-τ * Fair_i)
```

其中：
- `g_i`: 客户端i的梯度更新
- `g_s`: 服务器参考模型的梯度
- `Fair_i`: 客户端i的公平性偏差（在服务器数据上评估）
- `τ`: 平衡系数

**关键组件**:
1. 服务器参考模型（在干净数据上训练）
2. 余弦相似度计算（效用偏差）
3. 公平性评估（公平性退化）
4. 信任评分计算
5. 自适应聚合（只聚合高信任分数的客户端）

#### 4.2 实现步骤

**步骤1: 创建GuardFed.py**
```bash
touch src/algorithms/GuardFed.py
```

**步骤2: 实现Client类**
```python
class Client:
    def __init__(self, client_id, data, sensitive_features,
                 batch_size, learning_rate, model_class,
                 input_size, attack_form=None):
        # 基本初始化（类似FLTrust Client）

    def local_train_guardfed(self, global_model):
        # 本地训练逻辑
        # 返回模型更新
```

**步骤3: 实现Server类**
```python
class Server:
    def __init__(self, global_model, clients, algorithm, hyperparams,
                 server_data=None, tau=1.0):
        self.global_model = global_model
        self.clients = clients
        self.server_data = server_data
        self.tau = tau  # 平衡系数

        # 初始化服务器参考模型
        self.server_model = copy.deepcopy(global_model)

    def train_server_model(self):
        """在服务器数据上训练参考模型"""
        # 使用server_data训练server_model

    def compute_trust_scores(self, client_updates):
        """计算双视角信任评分"""
        trust_scores = {}

        # 1. 训练服务器参考模型，获取g_s
        server_update = self.train_server_model()

        for client_id, client_update in client_updates.items():
            # 2. 计算余弦相似度（效用偏差）
            cos_sim = self.compute_cosine_similarity(
                client_update, server_update
            )
            cos_sim = max(0, cos_sim)  # ReLU

            # 3. 计算公平性偏差
            fairness_deviation = self.compute_fairness_deviation(
                client_id, client_update
            )

            # 4. 计算信任分数
            trust_scores[client_id] = cos_sim * np.exp(-self.tau * fairness_deviation)

        return trust_scores

    def compute_cosine_similarity(self, update1, update2):
        """计算两个更新的余弦相似度"""
        vec1 = torch.cat([v.flatten() for v in update1.values()])
        vec2 = torch.cat([v.flatten() for v in update2.values()])
        cos_sim = F.cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(0))
        return cos_sim.item()

    def compute_fairness_deviation(self, client_id, client_update):
        """在服务器数据上评估客户端更新的公平性"""
        # 1. 将客户端更新应用到临时模型
        temp_model = copy.deepcopy(self.global_model)
        temp_model.load_state_dict(client_update)

        # 2. 在服务器数据上评估公平性
        fairness_metrics = self.evaluate_fairness_on_server_data(temp_model)

        # 3. 计算公平性偏差（AEOD或ASPD）
        fairness_deviation = fairness_metrics['EOD']  # 或者使用SPD

        return abs(fairness_deviation)

    def evaluate_fairness_on_server_data(self, model):
        """在服务器数据上评估模型公平性"""
        # 使用server_data评估模型
        # 返回公平性指标（EOD, SPD等）

    def aggregate(self, client_updates, trust_scores, threshold=0.5):
        """自适应聚合：只聚合高信任分数的客户端"""
        # 1. 筛选高信任分数的客户端
        selected_clients = [
            client_id for client_id, score in trust_scores.items()
            if score > threshold
        ]

        # 2. 归一化权重（类似FLTrust）
        normalized_updates = {}
        server_update = self.train_server_model()
        server_norm = torch.norm(torch.cat([v.flatten() for v in server_update.values()]))

        for client_id in selected_clients:
            client_update = client_updates[client_id]
            client_norm = torch.norm(torch.cat([v.flatten() for v in client_update.values()]))
            scale = server_norm / (client_norm + 1e-10)
            normalized_updates[client_id] = {
                k: v * scale for k, v in client_update.items()
            }

        # 3. 加权聚合
        total_trust = sum(trust_scores[c] for c in selected_clients)
        aggregated = {}
        for key in self.global_model.state_dict().keys():
            aggregated[key] = sum(
                trust_scores[c] / total_trust * normalized_updates[c][key]
                for c in selected_clients
            )

        # 4. 更新全局模型
        self.global_model.load_state_dict(aggregated)

    def run_round(self, round_num, test_df, y_test_values, model_class):
        """运行一轮训练"""
        # 1. 客户端本地训练
        client_updates = {}
        for client in self.clients:
            update, loss = client.local_train_guardfed(self.global_model)
            client_updates[client.client_id] = update

        # 2. 计算信任分数
        trust_scores = self.compute_trust_scores(client_updates)

        # 3. 自适应聚合
        self.aggregate(client_updates, trust_scores)

        # 4. 评估
        # ...
```

**步骤4: 测试验证**
```python
# 快速测试（1轮，1个攻击）
python scripts/test_guardfed.py

# 完整测试（50轮，5个攻击）
python scripts/run_guardfed_full.py
```

**预计时间**: 2-3天

---

## 📅 时间线

| 阶段 | 任务 | 预计时间 | 优先级 |
|------|------|---------|--------|
| 阶段1 | 添加非RW版本支持 | 0.5天 | ⭐⭐ |
| 阶段2 | 验证/实现FairGuard | 0.5-1天 | ⭐⭐⭐ |
| 阶段3 | 实现Hybrid | 1天 | ⭐⭐ |
| 阶段4 | 实现GuardFed | 2-3天 | ⭐⭐⭐⭐⭐ |
| 测试 | 完整实验 | 2-3小时 | - |
| **总计** | | **4-6天** | - |

---

## 🎯 实施顺序建议

### 方案A: 按优先级（推荐）
1. **GuardFed** (最重要) - 2-3天
2. **Hybrid** - 1天
3. **FairGuard** - 0.5-1天
4. **非RW版本** - 0.5天

**优点**: 先完成最重要的算法
**缺点**: 非RW版本最后才有

### 方案B: 按难度
1. **非RW版本** (最简单) - 0.5天
2. **FairGuard** - 0.5-1天
3. **Hybrid** - 1天
4. **GuardFed** (最复杂) - 2-3天

**优点**: 从简单到复杂，逐步推进
**缺点**: 最重要的算法最后才完成

### 方案C: 并行（如果有多人）
- 人员1: GuardFed
- 人员2: Hybrid + FairGuard
- 人员3: 非RW版本

**优点**: 最快完成
**缺点**: 需要多人协作

---

## 📋 详细任务清单

### Task 1: 添加非RW版本支持 ✓
- [ ] 1.1 修改 `src/algorithms/FedAvg.py`
  - [ ] Client类添加 `use_reweighting` 参数
  - [ ] 修改 `local_train_fedavg` 方法
- [ ] 1.2 修改 `src/algorithms/FairFed.py`
  - [ ] Client类添加 `use_reweighting` 参数
  - [ ] 修改训练方法
- [ ] 1.3 修改 `src/algorithms/Medium.py`
  - [ ] Client类添加 `use_reweighting` 参数
  - [ ] 修改训练方法
- [ ] 1.4 修改 `src/algorithms/FLTrust.py`
  - [ ] Client类添加 `use_reweighting` 参数
  - [ ] 修改训练方法
- [ ] 1.5 更新实验脚本
  - [ ] 添加非RW版本的实验配置
- [ ] 1.6 测试验证
  - [ ] 快速测试（1轮）
  - [ ] 对比RW vs 非RW结果

### Task 2: 验证/实现FairGuard ✓
- [ ] 2.1 检查FairG.py实现
  - [ ] 阅读代码逻辑
  - [ ] 确认是否是FairGuard
- [ ] 2.2 如果FairG = FairGuard
  - [ ] 创建FairGuard算法（复用FairG）
  - [ ] 测试验证
- [ ] 2.3 如果FairG ≠ FairGuard
  - [ ] 查找FairGuard论文
  - [ ] 实现FairGuard
  - [ ] 测试验证

### Task 3: 实现Hybrid ✓
- [ ] 3.1 创建 `src/algorithms/Hybrid.py`
- [ ] 3.2 实现HybridClient类
- [ ] 3.3 实现HybridServer类
  - [ ] 整合FLTrust信任评分
  - [ ] 整合FairGuard公平性筛选
  - [ ] 实现组合聚合逻辑
- [ ] 3.4 测试验证
  - [ ] 快速测试（1轮）
  - [ ] 完整测试（50轮）

### Task 4: 实现GuardFed ✓
- [ ] 4.1 阅读论文Section 5
  - [ ] 提取算法伪代码
  - [ ] 理解信任评分公式
  - [ ] 确定超参数
- [ ] 4.2 创建 `src/algorithms/GuardFed.py`
- [ ] 4.3 实现GuardFedClient类
- [ ] 4.4 实现GuardFedServer类
  - [ ] 服务器参考模型训练
  - [ ] 余弦相似度计算
  - [ ] 公平性偏差计算
  - [ ] 信任评分计算
  - [ ] 自适应聚合
- [ ] 4.5 测试验证
  - [ ] 快速测试（1轮，1攻击）
  - [ ] 中等测试（10轮，3攻击）
  - [ ] 完整测试（50轮，5攻击）
- [ ] 4.6 调优
  - [ ] 调整τ参数
  - [ ] 调整信任阈值
  - [ ] 对比论文结果

### Task 5: 运行完整实验 ✓
- [ ] 5.1 更新 `scripts/run_table4_final.py`
  - [ ] 添加所有新算法
  - [ ] 配置实验参数
- [ ] 5.2 运行实验
  - [ ] IID分布
  - [ ] non-IID分布
  - [ ] 所有攻击类型
- [ ] 5.3 保存结果
  - [ ] JSON格式
  - [ ] 表格格式

### Task 6: 结果验证和分析 ✓
- [ ] 6.1 对比论文表4
  - [ ] 数值对比
  - [ ] 趋势对比
- [ ] 6.2 分析差异
  - [ ] 找出差异原因
  - [ ] 调整参数
- [ ] 6.3 生成最终表格
  - [ ] LaTeX格式
  - [ ] Markdown格式

---

## 🚀 立即开始

### 建议: 从GuardFed开始（方案A）

**理由**:
1. GuardFed是论文核心，最重要
2. 实现过程中会加深对整个系统的理解
3. 其他算法相对简单，可以快速补充

**第一步**: 仔细阅读论文Section 5，提取GuardFed的算法细节

**第二步**: 创建 `src/algorithms/GuardFed.py`，开始实现

**第三步**: 边实现边测试，确保每个组件正确

---

## ❓ 需要确认

1. **是否按方案A的顺序实施？**（先GuardFed）
2. **τ参数的默认值是多少？**（需要从论文中确定）
3. **服务器数据比例用多少？**（论文中测试了1%-10%）
4. **是否需要我现在开始实现GuardFed？**
