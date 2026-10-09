# FLGMM 完整覆盖 accepted3 后单次增量

本次只采原collector唯一快照，实际2新终轮：IID Benign seed91005/91006。原strict、source/data前后、配置、70round、真实alpha、valid19867、root16277、同checkpoint全部指标与源身份通过；原本机verifier完整member/原checked_result通过。累计新增训练严格离机5/96，另4原70轮复用，不能称完整100已完成。

原collect_delta.py及verify_delta_offserver.py字节完整复制，科学loop未改；远端独占本目录名，CPU106单线程nice10/idle/CUDA隐藏，执行结束已释放。只有dispatch命名空间/parent guard与previous凭据作外层适配。上一OFFSERVER精确SHA bdb72d40fdb99a582ab0e6b346900cd92202611eedb25692c4ba05c91fe7b333；未initial-empty，未重包旧模型或追后到终轮。无CNN/test/训练/方法/服务/主链变化。

服务器实际接受torch2.11 cu128；本机torch2.8 CPU只读原保存模型/记录/张量及哈希，不声称同一runtime或新推理。保留所有原结果和负结果。

## 根只读采用入口

```powershell
Get-Content tmp/celeba_flgmm_fullcoverage_delta_after3_20261010/ROOT_READY_HANDOFF.json
Get-Content tmp/celeba_flgmm_fullcoverage_delta_after3_20261010/batch/OFFSERVER_ACCEPTANCE.json
Get-FileHash -Algorithm SHA256 tmp/celeba_flgmm_fullcoverage_delta_after3_20261010/batch/accepted_delta.tar.gz
```

实际原工具命令/exit0记录在 VERIFY_COMMAND.json（已运行，不应在现存restored目录盲重复）。根独立核 DELIVERY_FILES_SHA256 与MEMBERS/archive后另行写根采用；后续collector的--previous应指向本目录batch/OFFSERVER_ACCEPTANCE.json，--previous-sha256为该实际proof SHA，不能用初始空链。此交付没有更改LATEST_BACKUP、STATE、RUNNING或Git；根采用仍待根执行。
