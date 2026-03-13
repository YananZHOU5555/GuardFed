# GuardFed 表4复现 - 修复总结

## 状态: ✅ 所有错误已修复

**测试结果**: 35/35 成功 (100%)

---

## 修复的问题

### 1. 全局模型同步问题
**症状**: 训练后准确率始终为24%，权重不变
**原因**: Server类使用deepcopy，外部模型未同步
**修复**:
```python
# 在每轮训练后添加
global_model.load_state_dict(server.global_model.state_dict())
```
**文件**: `scripts/test_final.py`, `scripts/run_table4_final.py`

### 2. A_PRIVILEGED 未定义
**症状**: F Flip 和 S-DFA 攻击报错 "name 'A_PRIVILEGED' is not defined"
**原因**: 攻击代码使用了未定义的常量
**修复**: 在所有算法文件开头添加
```python
A_PRIVILEGED = 1  # Male
A_UNPRIVILEGED = 0  # Female
```
**文件**: `src/algorithms/FedAvg.py`, `FLTrust.py`, `FairFed.py`, `Medium.py`

### 3. FLTrust server_data 维度错误
**症状**: "mat1 and mat2 shapes cannot be multiplied (256x14 and 13x16)"
**原因**: server_data包含sex列，导致特征数为14而非13
**修复**:
```python
# 修改 FLTrust.py 第475行
server_X_tensor = torch.tensor(server_data_copy.drop(['income', 'sex'], axis=1).values, ...)
```
**文件**: `src/algorithms/FLTrust.py`

### 4. 全局变量设置
**修复**: 设置所有必需的全局变量
- HYPERPARAMETERS
- DEVICE
- MALICIOUS_CLIENTS
- test_inference_modified
- test_loader
- scaler
- numerical_columns

---

## 测试验证

运行 `python scripts/check_all_errors.py` 验证所有组合:

```
成功: 35/35
失败: 0/35

✅ FedAvg_RW + Benign/F Flip/FOE/S-DFA/Sp-DFA
✅ FairFed_RW + Benign/F Flip/FOE/S-DFA/Sp-DFA
✅ Medium_RW + Benign/F Flip/FOE/S-DFA/Sp-DFA
✅ FLTrust_RW + Benign/F Flip/FOE/S-DFA/Sp-DFA
✅ FedAvg_RW_FairG + Benign/F Flip/FOE/S-DFA/Sp-DFA
✅ FLTrust_RW_FairG + Benign/F Flip/FOE/S-DFA/Sp-DFA
✅ FedAvg_RW_FairCosG + Benign/F Flip/FOE/S-DFA/Sp-DFA
```

准确率: 75-76% (合理范围)

---

## 运行完整实验

```bash
python scripts/run_table4_final.py
```

**配置**:
- 50轮训练
- 70个实验 (7算法 × 5攻击 × 2分布)
- 结果保存到 `results/table4_final.json`

**预计时间**: 2-3小时

---

## 注意事项

**Sp-DFA 攻击**: 当前映射为'mixed'，代码中显示"未执行任何攻击"。测试通过但攻击未生效。如果论文需要Sp-DFA结果，需要检查原始论文定义并实现该攻击。

---

## 修改的文件

1. `src/algorithms/FedAvg.py` - 添加常量
2. `src/algorithms/FLTrust.py` - 添加常量 + 修复server_data
3. `src/algorithms/FairFed.py` - 添加常量
4. `src/algorithms/Medium.py` - 添加常量
5. `scripts/test_final.py` - 添加模型同步
6. `scripts/run_table4_final.py` - 添加模型同步
7. `scripts/check_all_errors.py` - 新建错误检查脚本

---

## 快速验证命令

```bash
# 快速测试 (3轮, 2个实验)
python scripts/test_final.py

# 检查所有组合 (1轮, 35个实验)
python scripts/check_all_errors.py

# 完整实验 (50轮, 70个实验)
python scripts/run_table4_final.py
```
