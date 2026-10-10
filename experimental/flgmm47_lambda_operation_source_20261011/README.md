# Fixed-operand lambda operation probe — prepared only

Independent read-only review verified diagnostic report `e2aee292…` and its measured object `086f43da…`. Only shared-calibration `server_adaptive_lambda` and its dependent fit hash differ. Windows lambda is `0x1.79a3d4dd52843p-1`; saved Linux lambda is `0x1.79a3d4dd52844p-1`. Thresholds, root receipt, all three prediction vectors, metrics and counts match. Both original scientific failure `c93d11ee…` and first diagnostic comparator failure `cbc5d0b5…` remain pinned. This does not convert the failed Windows check to acceptance.

Original core `cdd55865…`, lines 210–224, computes:

```python
base_risk = 0.5 * (aeod + aspd)
temp = max(config.ad2_calibration_temperature, 1e-6)
adaptive_lambda = config.ad2_calibration_base_weight * math.log1p(math.exp((base_risk - config.ad2_calibration_budget) / temp))
```

The original path calls neither `sum` nor `fsum`. Both captured base metrics and base risk already match exactly. `INPUTS.json` includes complete concrete decimal/hex operands, exact original expression/source fragments, checkpoint/cache/receipt identities, runtime observations and evidence pins. The probe includes explicitly labeled `sum` and `fsum` controls; they are not substitutions in the experiment.

After source review, root can run the same small sealed package in each existing runtime and retain stdout:

```text
python -B operation_probe.py --source-seal-sha256 <FILES_SHA256_SHA>
```

It traces binary addition, builtin sum, fsum, risk, subtraction, division, exp, log1p and final multiplication, with each intermediate's exact float hex. A separate branch starts directly from the captured common risk to exclude upstream metric computation. No NumPy, Torch, data arrays, root fitting, CNN, test, SSH, environment installation or adoption occurs. Preparation only compiled the source and checked operand/source identities; the operation probe has not run.

Concrete causal test: compare the first differing intermediate under the two actually used runtimes (Windows Python 3.10.4 versus Linux Python 3.12.3). Matching subtraction/division followed by a first difference in exp or log1p would localize this arithmetic discrepancy to that operation in those measured runtimes; it would not by itself distinguish interpreter version, operating-system library, compiler or another runtime cause. Only after that result should root decide whether a separate offserver Python 3.12 runtime adds useful discrimination. Do not install a runtime, refit 47 records, change tolerance or relax acceptance merely to obtain a match.
