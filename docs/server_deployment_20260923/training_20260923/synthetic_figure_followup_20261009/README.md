# ForestDiffusion 来源追加检查

2026-10-09 对已授权服务器 `89.22.197.55:60350` 的六个现有 GuardFed 项目目录做只读有限检索。631 个源码/依赖声明候选未找到 ForestDiffusion 名称；这些项目的常规虚拟环境目录及 `/opt/venv`、`/venv` 也未找到同名安装目录。没有导入历史代码、安装依赖、推理或训练。

原始范围、时间与结果见 `remote_bounded_search.json`；摘要及原字节 SHA 见 `receipt.json`。本次排除 data/results/outputs、Git、虚拟环境源码和缓存；虚拟环境仅检查常规 site-packages 顶层同名包。没有搜索其他个人目录、其他项目或不可达实例。

这补充了此前本地归档/Git 检索，仍无法恢复当时执行的 adapter、依赖、拟合模型和缓存身份，也不能据此断言历史实验从未执行 ForestDiffusion。原260条数值追溯及其封存证明保持不变；该实现身份缺口继续保留，不用新下载的库冒充历史版本。
