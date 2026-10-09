# FLGMM seven-canary launch operations — PREPARED, NOT EXECUTED

This independent operations envelope launches only the existing, unchanged `run_canaries.py`: two original three-round reference runs and five new three-round attack-interface runs, sequentially. It never starts the 96 new seventy-round validation queue. No SSH, supervisor command, CNN or experiment has been executed while preparing this package.

## Actual bound inputs

- Remote stage: `/workspace/guardfed_checks/celeba_flgmm_fullcoverage_v2_20261009/stage`.
- Package SHA: `6200c40f22a8ea670cfa02f0e74bbd81dc4ef9adecc258f979ddacc078a5d230`.
- Actual root adoption: `tmp/celeba_flgmm_fullcoverage_binding_20261009/ROOT_BOUND_ADOPTION.json`, SHA `fcecfc0a3582695edfd54c70db38e7dafdd5bf46dcdff8212b9dfc03fc7506fc`.
- This adoption explicitly verifies the downloaded 144-member metadata archive. Supply that same real receipt as both bound-offserver and bound-root-review inputs; these are two bindings to one proof, not two independent verifications. Do not fabricate another receipt.
- The earlier local Python 3.10 `tar.extractall(filter=...)` failure remains preserved. Root manually verified/extracted the original archive once. This launcher transports small files via base64, checks SHA and writes exclusive filenames; it uses no tar extraction API and does not repeat metadata binding.

## External approval and command

Root creates an approval outside this package from `APPROVAL_TEMPLATE.json`, changes status only after review, and fills the actual helper seal and baseline SHA. The template itself is deliberately rejected. Root supplies all approval, source and input SHA values explicitly:

```powershell
python -B tmp/celeba_flgmm_fullcoverage_canary_operations_20261009/launch.py --approval ROOT_APPROVAL_PATH --approval-sha256 ACTUAL_APPROVAL_SHA --bound-offserver ROOT_BOUND_ADOPTION_PATH --bound-offserver-sha256 fcecfc0a3582695edfd54c70db38e7dafdd5bf46dcdff8212b9dfc03fc7506fc --bound-root-review ROOT_BOUND_ADOPTION_PATH --bound-root-review-sha256 fcecfc0a3582695edfd54c70db38e7dafdd5bf46dcdff8212b9dfc03fc7506fc --baseline ACTUAL_ROOT_LIVE_PATH --baseline-sha256 ACTUAL_ROOT_LIVE_SHA --package-sha256 6200c40f22a8ea670cfa02f0e74bbd81dc4ef9adecc258f979ddacc078a5d230 --helper-seal-sha256 ACTUAL_FILES_SHA256_SHA
```

No approval is created by this preparation. The baseline must use the existing root-live schema (`checked_utc`, `queue_completed`, `active[].id/progress.round`), be at most one hour old, and show subsequent main round or completion growth at launch. The independently collected resource receipt must be at most 120 seconds old at supervisor start. The authorization is external to the bound package and is rejected by the fullcoverage scope.

## Launch boundary and evidence

Only `guardfed_celeba_flgmm_fullcoverage_canary` is installed/updated/started; autostart=false, autorestart=false, startretries=0. Existing main, Hybrid and other services are untouched. The old FL screen must be EXITED and sglang STOPPED. No prior stage authorization, canary output, service configuration, failed operations namespace or producer is reused. Restricted task/thread affinity owners may not overlap CPUs 102/103. Affinity must actually permit these CPUs.

The manager and sequential child use CPUs 102/103, nice 10, idle I/O and computation libraries set to one thread. The original authorization schema requires max_workers=2, but this gate runs one child at a time, not two parallel GPU workers. Declared CPU budgets include all observed Python owners plus three conservative slots (child, coordinator and launching helper reserve); auxiliary threads are recorded individually, not treated as full CPU cores. Real cgroup v2 quota/memory/events, both GPU free memory/recovery, disk, protected main queue/worker/round/log state and exact source/data mapping hashes are collected before starting.

`main_health.py` preserves the existing reviewed `need` and `main_health` function AST/source segments. Main nominal concurrency stays eight; actual 1–8 workers and completion growth allow normal terminal handover. The launcher does not equate RUNNING or completed counters with scientific acceptance.

The source lineage and 25 local checks are in `HEALTH_LINEAGE.json` and `SELF_CHECK.json`. Local tests exercised the real authorization prefix to its transport boundary, twelve wrong-approval refusals, original healthy 1/8-worker cases, three health refusals and optimization rejection. No Linux resource/supervisor behavior was simulated as real success. An initial selfcheck typo (`source_path` instead of the actual lineage key `source`) was corrected before passing; it made no remote or scientific change.

Remote operation failures and command receipts remain in the unique operations directory. Timeout after transfer/start is an uncertain operation, not permission to rerun: inspect the precise service and original namespace first. Existing stage paths and authorization make blind repetition fail. Startup receipts report only start-command success, never completed or scientifically accepted canaries. For subsequent read-only inspection, root may reuse the existing `celeba_flgmm_fullcoverage_root_operations_20261009/observe.py`; no new monitor is installed.

The selected recipe retains Tg=20; three rounds cannot exercise its later UCL/monitor behavior or prove seventy-round equivalence. The exact scientific comparison and all result acceptance remain in the original bound canary code, unchanged. Fullcoverage needs separately reviewed actual gate acceptance and a distinct scope authorization.
