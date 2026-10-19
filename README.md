# GuardFed — 导师代码交付

公平性感知的拜占庭鲁棒联邦学习研究代码。此分支整理了表格数据与 CelebA 图像实验的核心、基线适配器、评价及回归检查，便于阅读和交接。

源码固定自研究分支提交 [`37c35d6`](https://github.com/YananZHOU5555/GuardFed/tree/37c35d6c954d0a94e4221c378f9405a278c59df5)。训练算法、攻击、参数默认值和原有源码字节均保留；逐文件来源与校验值见 [SOURCE_SELECTION.json](SOURCE_SELECTION.json)。原始结果、模型、数据、大型缓存及历史工作记录保留在原研究材料中。

## 从哪里读

| 内容 | 入口 |
|---|---|
| AD2 / AD2+、候选选择、攻击、训练、指标及校准 | [scripts/reproduce_paper_tables.py](scripts/reproduce_paper_tables.py) |
| Adult / COMPAS 数据预处理与分区 | [src/data_loader.py](src/data_loader.py) |
| CelebA CNN 与图像数据适配 | [src/celeba_data.py](src/celeba_data.py) |
| 单项冻结任务、终轮检查、模型校验 | [scripts/run_revision_ablation.py](scripts/run_revision_ablation.py) 的 `worker --job` |
| FedAA / LASA / FLGMM / NGA 等研究适配代码 | [experimental/celeba_baselines](experimental/celeba_baselines) |
| 共用校准和部署参考 | [deployment](deployment) |
| 运行说明、数据位置与检查命令 | [docs/REPRODUCING.md](docs/REPRODUCING.md) |
| 方法对应关系及适配限制 | [docs/CODE_MAP.md](docs/CODE_MAP.md) |

`src/algorithms/` 保留早期模块供追溯；当前 AD2+ 主实现位于上表第一项。目录内历史 REPORT / AUDIT / PROTOCOL 文档保留各自日期和状态，不表示当前整体进度。

## 获取与安装

```bash
git clone --depth 1 --branch codex/advisor-code-20261019 https://github.com/YananZHOU5555/GuardFed.git
cd GuardFed
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/reproduce_paper_tables.py --help
python scripts/run_revision_ablation.py --help
```

Windows 激活环境使用 `.venv\Scripts\Activate.ps1`。Python 3.12 是历史服务器环境；本次入口及回归检查使用本机 Python 3.10.4。`configs/environment_5090.json` 是历史记录，`requirements.txt` 是依赖范围，二者都不是通用环境锁。

CelebA 缓存构建另需 `python -m pip install -r requirements-image.txt`。数据准备完成前先阅读运行说明；仓库不附数据、模型或结果。

## 交付边界

- 主命令行的 `--datasets` 只接受 Adult / COMPAS；CelebA 使用 JSON job 和共用 worker。
- `deployment/` 和部分研究适配器保留原服务器路径、冻结哈希及历史任务依赖，迁移条件见代码地图。
- LoGoFair 需要另取固定版本的官方源码与 `netcal`。方法名称出现在 CLI 中，不代表已完成原论文的忠实复现；以具体实现和适配说明为准。
- CelebA 既有成绩属于验证集证据，配置选择和历史测试集暴露需要披露；这份代码整理不证明最终独立测试或整项返修完成。
- 本次回归检查为 **11 passed / 1 skipped**；CUDA 确定性检查因本机无 CUDA 跳过，详见 [验证记录](docs/VERIFICATION.json)。未重新训练、调参或生成论文成绩。

研究材料可从[原研究分支](https://github.com/YananZHOU5555/GuardFed/tree/codex/revision-evidence-baselines-20260928)追溯。第三方代码保留原有来源及许可证文件；本次交付不新增统一开源许可证。
