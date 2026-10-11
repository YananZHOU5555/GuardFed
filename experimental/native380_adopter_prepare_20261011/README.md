# native380 adopter — source only

原370 adopter最小元数据派生；尚未执行、未根采用。Root先读SOURCE_DIFF和ROOT_SOURCE_REVIEW，核F卷Yanan 2TB/Healthy/容量及canonical ledger仍为4baa91…，再将ACTUAL_INPUTS.template.json原字节复制成ACTUAL_INPUTS.json，仅经root授权后把root_execution_authorized改true，计算实际SHA。唯一执行入口：

```powershell
python -B tmp/native380_adopter_prepare_20261011/adopt_root.py --actual-inputs-sha256 <ROOT实际授权的ACTUAL_INPUTS.json SHA256>
```

新批为non-IID FedSA91001..10，实际原strict/offserver110成员；archive/member/model与receipt检查循环原字节不变。旧370 inspection保留471对象；其中FedSA91001为已记录pending，只从旧接受投影排除，470对象（Full100+旧native370）及顺序精确保持。完整新inspection480对象，ledger51条前缀不变，追加1条至52。原远端/本地原子ledger提升逻辑仅namespace改变。实际归档仍在F，由root运行时读取；本次准备只读E元数据。任何失败保留且不自动重试。没有训练、forward、fit、test或新统计。
