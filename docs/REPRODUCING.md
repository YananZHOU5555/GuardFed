# 安装、数据与验证

## 安装检查

从仓库根目录执行：

```bash
python -m pip install -r requirements.txt
python scripts/reproduce_paper_tables.py --help
python scripts/run_attack_strength_study.py --help
python scripts/run_revision_ablation.py --help
python scripts/run_ad2plus_advisor_experiments.py --suite ablation --dry-run
```

这些命令只显示入口或计划。本次交付已检查其退出状态。`--smoke` 会加入多个攻击任务，不能当成安装检查。GPU 包应与实际机器匹配；这里没有重新安装或认证新的 CUDA 环境。

## 数据和输出位置

本交付不附数据。核心所需路径为：

```text
data/adult/adult.data
data/adult/adult.test
data/adult/adult.names
data/compas/compas-scores-two-years.csv
data/celeba/derived/rgb64_v1/{images.npy,metadata.npz,available.npy,manifest.json}
```

Adult / COMPAS 的固定原文件仍可从[来源提交的 data 目录](https://github.com/YananZHOU5555/GuardFed/tree/37c35d6c954d0a94e4221c378f9405a278c59df5/data)取得。校验值见 [DATA_SOURCES.json](DATA_SOURCES.json)。主 tabular CLI 会先校验两个数据集，即使只选 COMPAS，也需准备上列四个表格数据文件。CelebA 需要官方属性与 split 元数据及原图像分片；详情见 [图像管线参考](reference/CELEBA_PIPELINE.md)。

大型原始数据、图像缓存、模型和逐轮输出应放在科研存储盘或服务器。亚楠本机使用 `F:/YananResearchStorage/GuardFed/`，写入前确认 F 盘卷标为 `Yanan 2TB` 且容量足够；不可用时停止大型写入。项目 `data/`、`results/` 可用目录链接保持代码路径兼容。Linux 可在空目标上建立链接，例如：

```bash
GUARDFED_STORAGE=/path/to/research-storage/GuardFed
mkdir -p data
ln -s "$GUARDFED_STORAGE/data/adult" data/adult
ln -s "$GUARDFED_STORAGE/data/compas" data/compas
ln -s "$GUARDFED_STORAGE/data/celeba" data/celeba
ln -s "$GUARDFED_STORAGE/results" results
```

先准备存在的存储目录；已有 data / results 目录应先核对内容，不覆盖。图像缓存约数 GB，不随此包分发。

缓存构建使用独立环境或补装图像预处理依赖：

```bash
python -m pip install -r requirements-image.txt
python scripts/build_celeba_cache.py --root data/celeba --workers 8
```

脚本读取 `list_attr_celeba.txt`、`list_eval_partition.txt` 与 `hf_flwrlabs/img_align+identity+attr/` 下的 Parquet 图像分片，按官方图像编号构建缓存。它不会替你下载数据；需要完整 split / 属性与图像身份检查。

## 回归检查

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q tests
```

其中三组测试用 `git show` 读取历史提交 `ec419b7` / `9e62b78`，COMPAS 检查还需要对应 CSV。浅克隆先执行 `git fetch --unshallow origin`；完整历史包含旧研究材料，体积较大，按需在服务器或外置盘操作。ZIP 没有 Git 历史，只适合阅读及运行不依赖历史的检查。

本次在包含历史和既有数据的来源工作区运行四个测试文件：11 通过，CUDA median 检查 1 跳过。在导出包内单独运行同一 CUDA 检查亦跳过。导出源码逐文件与远端 Git blob 相符；未据此宣称全部基线或新机器 GPU 复现通过。

## 表格数据的一轮 CPU 管线检查（可选）

准备好数据和外部输出目录后，以下命令会真正训练并写入 `results/paper_tables/`：

```bash
python scripts/reproduce_paper_tables.py --full --rounds 1 --device cpu \
  --datasets compas --distributions IID --methods GuardFed-AD2+ \
  --attacks Benign --ad2-plus-mode fixed \
  --experiment-suite installation_check --experiment-tag cpu_1round
```

这只有一轮，代码内部仍使用 `full` 标签；不构成论文实验。本次交付未执行此训练命令。入口末尾可能报告完整论文表缺项，这与单项管线检查是不同检查范围。

## CelebA 和正式实验

CelebA 使用 `scripts/run_revision_ablation.py worker --job /path/to/job.json`。job 必须显式给出 dataset / distribution / method / attack、完整 `ExperimentConfig`、输出位置与源码哈希；图像配置必须显式指定 `celeba_evaluation_split="valid"`，因为旧 dataclass 的默认值是 `test`。不要把主 CLI 的 tabular 默认值当作图像正式配方。

`source_hashes()` 还会读取 Adult 的 train/test 文件以校验源码包身份；图像 worker 需要完整缓存和官方元数据。选择参数、开发种子、正式共享种子和评价 split 以对应冻结协议为准。本包未收入批量历史 job，也未创建新的正式协议。

历史攻击校准需同时保留 `configs/attack_strength_5090.json` 与 `configs/calibration_results_5090.jsonl`。续跑工具需要把二者恢复为目标结果目录下的匹配记录；只恢复配置可能触发重新校准。不能把 5090 历史校准自动当作新的图像正式校准。

CTGAN / TVAE 路径另需原阶段的 `ctgan` 环境；LoGoFair 需官方源码与 `netcal`。这些可选路径未纳入本次端到端执行。已有 CelebA 结果属于验证集证据，历史测试暴露及配置选择不能被代码整理消除。
