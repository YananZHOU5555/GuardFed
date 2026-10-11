# LoGo fixed2 start and observation source

Source only; actual deployment seal, queue auth, fresh observation and authorization remain null. Root may bind this template only after the v2 metadata installer actually succeeds. The installer owns artifact/cache/environment/source adoption; this start source verifies its exact actual seal and runs the unchanged original resource guard on CPU113 before any supervisor mutation.

The original FL100 supervisor pattern is reused: broad-affinity manager, only original guard/worker children use CPU113, nice10/idle, CUDA hidden, single-thread environment, no autorestart/start retries. Six project services permit RUNNING/EXITED as resource facts; sglang must stay STOPPED per deployed original queue auth. Root binds exact observed states separately; startup checks them against a fresh actual status snapshot, then original guard checks policy again. Existing attempts/service/config/namespace processes are refused. Already deployed BASE is required, not recreated. Existing canary/tasks/runs/lock/start markers are refused.

```
python run_root.py start --root-inputs ACTUAL.json --root-inputs-sha256 ACTUAL --output NEW_LOCAL_COMMAND_DIR
python run_root.py observe --output NEW_LOCAL_OBSERVATION_DIR
```

Remote source is sent via stdin, preventing source strings containing target IDs from polluting process argv during the guard scan. Full local stdout/stderr/exit/exception and actual expanded-source SHA are retained. Supervisor command results are printed before return-code assertions and saved on the server once mutation starts. Failure is preserved and never retried automatically. The read-only observer reads fixed2 JSON and bounded log tails, no arrays, models, pickle or scientific imports. It reports saved accepts with root_adopted=false; service EXITED/COMPLETE is not root scientific acceptance. Remaining98 is never started here.

No SSH, F access, installation, deployment, scientific execution or shared state change occurred while preparing this source. No new scientific source or resource guard is introduced. Original input/worker/fit/math/threshold/tolerance remains frozen. Actual pickle compatibility still depends on the root-run fixed2 canaries.
