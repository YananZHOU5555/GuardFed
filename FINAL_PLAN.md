# GuardFed 表4 - 最终实现计划

## 🎯 算法映射关系（重要！）

### 已明确的映射
1. **FairG = FairGuard** ✓
2. **GuardFed = FedAvg_RW_FairCosG** ✓（已训练完成）
3. **FairGuard = FedAvg_FairG** ✓（需要无RW版本）

### 论文表4需要的算法

| 表4中的名称 | 实际实现 | 状态 |
|------------|---------|------|
| FedAvg | FedAvg (无RW) | ❌ 需要实现 |
| FairFed | FairFed (无RW) | ❌ 需要实现 |
| Median | Median (无RW) | ❌ 需要实现 |
| FLTrust | FLTrust (无RW) | ❌ 需要实现 |
| FairGuard | FedAvg_FairG (无RW) | ❌ 需要实现 |
| FLTrust + FairGuard | FLTrust_FairG (无RW) | ❌ 需要实现 |
| GuardFed | FedAvg_RW_FairCosG | ✅ 已完成 |

---

## 📊 当前状态

### ✅ 已训练完成（7个）
1. FedAvg_RW ✓
2. FairFed_RW ✓
3. Medium_RW ✓
4. FLTrust_RW ✓
5. FedAvg_RW_FairG ✓
6. FLTrust_RW_FairG ✓
7. **FedAvg_RW_FairCosG** ✓ (= GuardFed)

### ❌ 需要补充（6个）
1. FedAvg (无RW)
2. FairFed (无RW)
3. Median (无RW)
4. FLTrust (无RW)
5. FedAvg_FairG (无RW) = FairGuard
6. FLTrust_FairG (无RW) = FLTrust + FairGuard

---

## 🔧 实施方案

### 方案：添加无RW版本支持

**核心思路**: 修改现有Client类，添加 `use_reweighting` 参数

**需要修改的文件**:
1. `src/algorithms/FedAvg.py`
2. `src/algorithms/FairFed.py`
3. `src/algorithms/Medium.py`
4. `src/algorithms/FLTrust.py`

---

## 📝 详细实施步骤

### 步骤1: 修改Client类支持无RW模式

#### 1.1 修改 FedAvg.py

```python
# src/algorithms/FedAvg.py

class Client:
    def __init__(self, client_id, data, sensitive_features, batch_size,
                 learning_rate, model_class, input_size, attack_form=None,
                 use_reweighting=True):  # 添加参数
        """
        初始化客户端类

        参数:
            use_reweighting: 是否使用Reweighting权重（默认True）
        """
        self.client_id = client_id
        self.X = data["X"]
        self.y = data["y"]
        self.sensitive_features = data["sensitive"]
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.attack_form = attack_form
        self.is_malicious = self.client_id in MALICIOUS_CLIENTS and self.attack_form != "no_attack"

        # 根据use_reweighting决定是否使用样本权重
        if use_reweighting:
            self.sample_weights = data.get("sample_weights", None)
        else:
            self.sample_weights = None  # 强制不使用权重

        # ... 其余初始化代码

    def local_train_fedavg(self, global_model):
        """
        执行本地训练（支持RW和非RW模式）
        """
        self.model.load_state_dict(global_model.state_dict())
        self.model.train()

        total_loss = 0.0
        total_batches = 0

        for epoch in range(HYPERPARAMETERS['LOCAL_EPOCHS']):
            for batch in self.train_loader:
                if self.sample_weights is not None:
                    # RW模式：使用样本权重
                    X_batch, y_batch, _, sample_weights_batch = batch
                    X_batch, y_batch = X_batch.to(HYPERPARAMETERS['DEVICE']), y_batch.to(HYPERPARAMETERS['DEVICE'])

                    self.optimizer.zero_grad()
                    logits = self.model(X_batch)
                    loss = self.criterion(logits, y_batch)

                    # 应用样本权重
                    sample_weights_batch = sample_weights_batch.to(HYPERPARAMETERS['DEVICE'])
                    loss = loss * sample_weights_batch
                    loss = loss.sum() / sample_weights_batch.sum()
                else:
                    # 非RW模式：不使用样本权重
                    X_batch, y_batch, _ = batch
                    X_batch, y_batch = X_batch.to(HYPERPARAMETERS['DEVICE']), y_batch.to(HYPERPARAMETERS['DEVICE'])

                    self.optimizer.zero_grad()
                    logits = self.model(X_batch)
                    loss = self.criterion(logits, y_batch)
                    loss = loss.mean()

                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                total_batches += 1

        avg_loss = total_loss / total_batches if total_batches > 0 else 0.0

        # 恶意客户端行为处理
        if self.is_malicious:
            if self.attack_form == "attack_acc_0.5":
                self.apply_weight_attack(HYPERPARAMETERS['W_attack_0_5'])
            elif self.attack_form == "attack_super_mixed":
                self.apply_weight_attack(HYPERPARAMETERS['W_attack_0_5'])

        return self.model.state_dict(), avg_loss
```

#### 1.2 修改 FairFed.py

类似的修改，添加 `use_reweighting` 参数

#### 1.3 修改 Medium.py

类似的修改，添加 `use_reweighting` 参数

#### 1.4 修改 FLTrust.py

类似的修改，添加 `use_reweighting` 参数

---

### 步骤2: 更新实验脚本

#### 2.1 创建新的实验脚本

```python
# scripts/run_table4_complete.py

# 算法列表（包含RW和非RW版本）
algorithms = [
    # 非RW版本（表4需要）
    'FedAvg',           # 无RW
    'FairFed',          # 无RW
    'Median',           # 无RW
    'FLTrust',          # 无RW
    'FedAvg_FairG',     # 无RW，即FairGuard
    'FLTrust_FairG',    # 无RW，即FLTrust+FairGuard

    # RW版本（已训练）
    'FedAvg_RW',
    'FairFed_RW',
    'Medium_RW',
    'FLTrust_RW',
    'FedAvg_RW_FairG',
    'FLTrust_RW_FairG',
    'FedAvg_RW_FairCosG',  # 即GuardFed
]

def run_single_experiment(algorithm, attack, alpha):
    """运行单个实验"""
    # 加载数据
    data_loader = DatasetLoader(...)

    # 确定是否使用RW
    use_reweighting = '_RW' in algorithm

    # 创建客户端
    clients = []
    for i in range(NUM_CLIENTS):
        # ...
        if 'FLTrust' in algorithm:
            client = FLTrustClient(
                i, client_data, sex_np,
                HYPERPARAMETERS['BATCH_SIZE'],
                HYPERPARAMETERS['LEARNING_RATES']['FLTrust_RW'][0],
                MLP, data_info['num_features'], attack_form,
                use_reweighting=use_reweighting  # 传递参数
            )
        else:
            client = FedAvgClient(
                i, client_data, sex_np,
                HYPERPARAMETERS['BATCH_SIZE'],
                HYPERPARAMETERS['LEARNING_RATES']['FedAvg_RW'][0],
                MLP, data_info['num_features'], attack_form,
                use_reweighting=use_reweighting  # 传递参数
            )
        clients.append(client)

    # 创建服务器
    # ...

    # 训练
    # ...
```

---

## 📋 任务清单

### Task 1: 修改算法文件 ✓
- [ ] 1.1 修改 `src/algorithms/FedAvg.py`
  - [ ] Client类添加 `use_reweighting` 参数
  - [ ] 修改 `local_train_fedavg` 方法
  - [ ] 测试RW和非RW模式

- [ ] 1.2 修改 `src/algorithms/FairFed.py`
  - [ ] Client类添加 `use_reweighting` 参数
  - [ ] 修改训练方法
  - [ ] 测试RW和非RW模式

- [ ] 1.3 修改 `src/algorithms/Medium.py`
  - [ ] Client类添加 `use_reweighting` 参数
  - [ ] 修改训练方法
  - [ ] 测试RW和非RW模式

- [ ] 1.4 修改 `src/algorithms/FLTrust.py`
  - [ ] Client类添加 `use_reweighting` 参数
  - [ ] 修改训练方法
  - [ ] 测试RW和非RW模式

### Task 2: 更新实验脚本 ✓
- [ ] 2.1 创建 `scripts/run_table4_complete.py`
  - [ ] 添加所有算法（RW和非RW）
  - [ ] 配置实验参数

- [ ] 2.2 创建快速测试脚本
  - [ ] `scripts/test_no_rw.py`
  - [ ] 测试非RW版本

### Task 3: 运行实验 ✓
- [ ] 3.1 快速测试（1轮）
  - [ ] 测试所有非RW版本
  - [ ] 验证结果合理性

- [ ] 3.2 完整实验（50轮）
  - [ ] 运行所有算法组合
  - [ ] IID和non-IID分布
  - [ ] 5种攻击类型

- [ ] 3.3 保存结果
  - [ ] JSON格式
  - [ ] 对比RW vs 非RW

### Task 4: 生成表4 ✓
- [ ] 4.1 整理结果
  - [ ] 提取所需算法的结果
  - [ ] 按表4格式组织

- [ ] 4.2 生成表格
  - [ ] LaTeX格式
  - [ ] Markdown格式

- [ ] 4.3 对比论文
  - [ ] 数值对比
  - [ ] 趋势分析

---

## ⏱️ 预计时间

| 任务 | 预计时间 |
|------|---------|
| 修改4个算法文件 | 2-3小时 |
| 更新实验脚本 | 1小时 |
| 快速测试 | 0.5小时 |
| 完整实验（50轮） | 2-3小时 |
| 生成表4 | 1小时 |
| **总计** | **6-8小时** |

---

## 🚀 立即开始

### 第一步: 修改FedAvg.py
```bash
# 编辑文件
code src/algorithms/FedAvg.py

# 添加use_reweighting参数
# 修改local_train_fedavg方法
```

### 第二步: 快速测试
```python
# 创建测试脚本
# 测试RW vs 非RW的差异
```

### 第三步: 批量修改其他文件
```bash
# 依次修改FairFed.py, Medium.py, FLTrust.py
```

### 第四步: 运行完整实验
```bash
python scripts/run_table4_complete.py
```

---

## 📊 最终表4算法列表

### COMPAS数据集
1. FedAvg (无RW)
2. FairFed (无RW)
3. Median (无RW)
4. FLTrust (无RW)
5. FairGuard = FedAvg_FairG (无RW)
6. FLTrust + FairGuard = FLTrust_FairG (无RW)
7. GuardFed = FedAvg_RW_FairCosG ✓

### ADULT数据集
同上

---

## ✅ 确认信息

根据您的说明：
- ✅ FairG = FairGuard
- ✅ GuardFed = FedAvg_RW_FairCosG（已完成）
- ✅ 所有算法文件都在 `src/algorithms/`
- ✅ 只需要添加无RW版本支持

**任务**: 修改现有算法，添加 `use_reweighting` 参数，然后重新训练无RW版本

**预计完成时间**: 6-8小时

---

## ❓ 需要确认

1. **是否现在开始修改FedAvg.py？**
2. **是否需要我提供完整的修改代码？**
3. **还是您希望我先创建测试脚本？**
