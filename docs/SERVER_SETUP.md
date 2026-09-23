# 服务器迁移与启动

本次源码来自 2026-09-23 保存的 5090 工作目录快照，包含较新的 AD2/AD2+、消融、根数据与攻击强度实验入口。它尚未逐项核验与 TDSC 投稿稿公式及实验协议一致；迁移成功不代表已经完成论文复现或返修实验。

发布版本只做运行路径、Python 解释器等迁移修复，并从编排脚本移除核心入口不支持的 `forest_diffusion` 选项；未重设攻击校准或研究协议。原始源码另存于 Release。`outputs/`、`.codex*` 中的旧材料仅供历史参考，不应直接覆盖当前源码或作为统一口径的最终结果。

## 1. 安装

以下命令面向 Linux，在计划存放项目的父目录执行：

```bash
git clone https://github.com/YananZHOU5555/GuardFed.git
cd GuardFed
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

建议使用 Python 3.12。5090 实际环境记录在 `configs/environment_5090.json`：Python 3.12.3、torch 2.12.0、numpy 2.4.6、pandas 2.3.3、scipy 1.17.1、scikit-learn 1.9.0。该文件是环境记录，不是跨机器通用的依赖锁；`requirements.txt` 的旧版本下限也不代表那些组合均经过验证。GPU 运行需选择与服务器驱动及显卡兼容的 PyTorch 安装包，并确认 CUDA 可用。

仅运行 CTGAN/TVAE 合成根数据实验时补装：

```bash
python -m pip install ctgan==0.12.1
```

原始数据随仓库提供：`data/adult/adult.data`、`adult.test`、`adult.names`，以及 `data/compas/compas-scores-two-years.csv`。加载路径由仓库位置推导，无需沿用 5090 的 `/home/...` 路径。

## 2. 先检查入口

以下命令只显示参数，不训练、不生成科研结果：

```bash
python scripts/reproduce_paper_tables.py --help
python scripts/run_attack_strength_study.py --help
python scripts/run_ad2plus_advisor_experiments.py --help
python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())"
```

主实验入口支持 `--device cpu`、`auto`、`cuda`。advisor 编排脚本的既有实验配置使用 CUDA；可先查看其将启动的命令：

```bash
python scripts/run_ad2plus_advisor_experiments.py --suite ablation --dry-run
```

不要把 `run_attack_strength_study.py --phase smoke` 当作单任务检查：它会先检查或运行校准，再运行多项实验。主入口的 `--smoke` 也会额外添加若干攻击任务。

## 3. 可选的单轮 CPU 检查

请在独立临时克隆执行下面的单任务检查，避免混入正式结果目录。它会写入 `results/paper_tables/`，且内部标记为 `full`；只有一轮，**不能作为论文结果或正式实验完成的证据**。

```bash
git clone https://github.com/YananZHOU5555/GuardFed.git ../GuardFed-migration-check
cd ../GuardFed-migration-check
python scripts/reproduce_paper_tables.py \
  --full --rounds 1 --device cpu \
  --datasets compas --distributions IID \
  --methods GuardFed-AD2+ --attacks Benign \
  --ad2-plus-mode fixed \
  --experiment-suite migration_smoke --experiment-tag cpu_1round
cd ../GuardFed
```

上面沿用已激活的虚拟环境。本次迁移已在独立 Windows 与 Linux 目录执行这项单轮检查，均成功；未启动正式实验。入口还会自动比较完整论文表，在空结果目录中报告 420 项缺失，这是未运行完整基准的预期结果，不是单轮检查失败。

## 4. 沿用既定攻击配置

在正式项目目录中，将已有配置与对应校准记录一起恢复。以下示例只补不存在的文件，保留已有运行记录：

```bash
mkdir -p results/attack_strength
test -e results/attack_strength/attack_config.json || cp configs/attack_strength_5090.json results/attack_strength/attack_config.json
test -e results/attack_strength/calibration_results.jsonl || cp configs/calibration_results_5090.jsonl results/attack_strength/calibration_results.jsonl
```

当前代码只有在 `attack_config.json` 存在且 `calibration_results.jsonl` 包含有效记录时才复用既有校准。**只复制配置文件会触发重新校准**。既定参数包括 `fflip_mode=all_unprivileged`、`fedsa_gain=4.5`、`fedsa_norm_ratio=3.0`；请勿为了迁移随意重新校准。若目标目录已有其他配置，先核对来源再决定使用哪份。

确认设备、协议、已有配置与结果目录后，可执行既有完整实验入口，例如：

```bash
python scripts/run_attack_strength_study.py --phase main --rounds 70 --device cuda
# 恶意客户端比例实验（单独运行）
python scripts/run_attack_strength_study.py --phase ratio --rounds 70 --device cuda
```

以上会真正训练，工作量较大；不是默认安装检查。它们延续 5090 的实验设计，不是已确定的新返修协议。脚本按已有运行标识跳过完成记录，断点续跑需要恢复相应原始结果。

## 5. 历史结果与完整快照

Release 标签 [`snapshot-5090-20260923`](https://github.com/YananZHOU5555/GuardFed/releases/tag/snapshot-5090-20260923) 保存：

- `guardfed-5090-20260923.tar.gz`：5090 原始源码及实验材料快照。
- `guardfed-windows-artifacts-20260923.tar.gz`：Windows 历史扩展材料与结果。

请先将压缩包解压到**独立恢复目录**，查看目录结构。不要直接在当前仓库根目录解压 5090 包，否则可能覆盖迁移后的源码。需要续跑时，仅复制所需的 `results/` 子目录，并保留原始 JSONL 与配置；若当前已经产生新结果，先核对重叠内容，避免覆盖。Windows 历史结果只用于核对来源，不应自动并入新的统计表。

现有不同脚本可能使用不同结果选轮和汇总规则。正式返修实验开始前，还需统一论文对应实现、数据划分、root 比例、攻击定义、随机种子和报告口径。

## 6. 外部基线源码

两个原始基线仓库以固定提交的子模块保存；当前统一实验入口无需导入它们。需要查看或独立运行原仓库时执行：

```bash
git submodule update --init --recursive
```

历史报告生成脚本部分依赖 `@oai/artifact-tool`，未打包本机 `node_modules`；训练与评估不依赖此报告工具。
