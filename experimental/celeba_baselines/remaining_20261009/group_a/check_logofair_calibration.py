"""Default calibrated official LoGoFair DP branch, isolated netcal environment."""
import hashlib
import json
import platform
from pathlib import Path

import netcal
import numpy as np
import torch

from adapters import fit_logofair

torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
rng = np.random.default_rng(7109)
cid = np.repeat(np.arange(20), 80)
group = np.tile(np.repeat([0, 1], 40), 20)
score = rng.uniform(.1, .9, len(cid)).astype(np.float32)
label = (rng.random(len(score)) < score).astype(int)
settings = dict(post_rounds=3, local_steps=3, global_steps=3, calibration=True)
post = fit_logofair(score, label, group, cid, **settings)
replay = fit_logofair(score, label, group, cid, **settings)
assert np.array_equal(post.predict(score, group, cid), replay.predict(score, group, cid))
assert all(torch.equal(post.thresholds[c], replay.thresholds[c]) for c in post.clients)
assert all(obj.calib for obj in post.clients.values())
assert len({tuple(t.tolist()) for t in post.thresholds.values()}) > 1
raw = fit_logofair(score, label, group, cid, **dict(settings, calibration=False))
assert any(not torch.equal(raw.thresholds[c], post.thresholds[c]) for c in post.clients)
all_calibrators = [cal for obj in post.clients.values()
                   for cal in [obj.score_calibrated_0, obj.score_calibrated_1]]
for cal in all_calibrators:
    calibrated = cal.transform(np.array([.2, .5, .8], dtype=np.float32))
    assert np.isfinite(calibrated).all() and ((calibrated >= 0) & (calibrated <= 1)).all()
report = {
    "status": "PASS", "evidence": "synthetic CPU scores, not real CelebA performance",
    "python": platform.python_version(), "torch": torch.__version__,
    "netcal": netcal.__version__, "numpy": np.__version__,
    "settings": settings, "clients": 20, "sample_count": len(cid),
    "per_group_beta_calibrators": len(all_calibrators),
    "checks": ["40 official BetaCalibration MLE fits", "all transformed probabilities finite",
               "deterministic replay of predictions and client thresholds", "calibration changes thresholds",
               "distinct client-specific group thresholds"],
    "global_lambda": post.global_lambda,
    "provenance": post.integration_provenance,
    "check_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
}
Path(__file__).with_name("logofair_calibration_gate.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, ensure_ascii=False, indent=2))
