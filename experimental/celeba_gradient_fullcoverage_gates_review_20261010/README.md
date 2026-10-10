# Independent gradient14 source review

结论：源码可供根继续准备，未发现实质问题；本审查不构成实际64闭合、192冻结、真实图像门通过或派发批准。

只读核验原封条8成员、40,350 bytes，原gate/共享cache注册、screen64和coverage源。原14个科学函数与6个嵌套数据/梯度/攻击oracle逐源码相同；aggregate oracle只改变FOE场景应调用root reference的预期。其余gate改动限于seed91002、三攻击root预期、明确攻击分配和探索阶段标记，正式70轮接受器没有放宽。

每方法7任务：F Flip/FedSA/Sp-DFA各screen与coverage，加Benign screen参考。共14个不同ID、6对实现比较、2对F Flip–Benign null比较；non-IID alpha=5、seed91002、batch64、3轮、CPU8线程、单fresh进程。原Benign/S-DFA四项门不重复。完整64原身份/已离机root和实际冻结192外部批准是binding必要条件，当前均未执行binding。

保存结果比较调用原strict检查，精确比较同checkpoint张量、三轮指标/diagnostics、native replay、梯度audit和指定RNG指纹。F Flip null仅忽略两种攻击标签，仍要求全部数值audit和raw/upload梯度SHA相等。该比较不新推理/拟合，科学表记录始终为0，不能替代70轮接受或GPU等价证明。

实际独立检查命令：`python -B tmp/celeba_gradient_fullcoverage_gates_review_20261010/review_once.py`（exit 0）。未调用作者check_prepared整套fixture、Torch、NumPy、服务器或数组。额外只核runtime metadata一项正向与14项拒收。`REVIEW.json`列出检查、原源SHA和局限；`FILES_SHA256.json`仅封本审查产物，原包未修改。

运行层局限：120秒资源凭据依赖根实际测量，源码本身不实时扫描CPU owner/GPU/IO；根仍须实际preflight、离机备份及单独采用。RNG只比较Python、NumPy全局与Torch CPU状态，不声称所有独立Generator。F Flip null来自原unweighted CE下Male仅元数据的实现，不代表对输入/标签攻击的鲁棒性。
