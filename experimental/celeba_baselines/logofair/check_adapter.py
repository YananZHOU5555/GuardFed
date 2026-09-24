"""Small CPU check; synthetic scores, not CelebA scientific results."""
import json
import numpy as np
import torch
from adapter import OfficialLoGoFairDP, COMMIT

torch.set_num_threads(1)
rng = np.random.default_rng(1234)
cid = np.repeat(np.arange(20), 80)
a = np.tile(np.repeat([0, 1], 40), 20)
p = rng.uniform(.05, .95, len(cid)).astype(np.float32)
p = np.clip(p + (a * 2 - 1) * (.03 + cid * .005), .01, .99)
y = (rng.random(len(p)) < p).astype(int)
settings = dict(post_rounds=3, local_steps=3, global_steps=3, calibration=False)
first = OfficialLoGoFairDP(**settings).fit(p, y, a, cid)
again = OfficialLoGoFairDP(**settings).fit(p, y, a, cid)
pred = first.predict(p, a, cid)
assert np.array_equal(pred, again.predict(p, a, cid))
assert set(np.unique(pred)) == {0, 1}
assert len(first.thresholds) == 20
assert len({tuple(t.tolist()) for t in first.thresholds.values()}) > 1
relaxed = OfficialLoGoFairDP(**settings, local_delta=.5).fit(p, y, a, cid)
assert any(not torch.equal(first.thresholds[c], relaxed.thresholds[c]) for c in first.clients)
global_relaxed = OfficialLoGoFairDP(**settings, global_delta=.5).fit(p, y, a, cid)
assert first.global_lambda != global_relaxed.global_lambda
try:
    first.predict([.7], [1], [999])
except ValueError:
    pass
else:
    raise AssertionError("Unknown evaluation client must be rejected")
try:
    OfficialLoGoFairDP(**settings).fit([.1,.9], [0,1], [0,0], [0,0])
except ValueError:
    pass
else:
    raise AssertionError("Missing client-sensitive stratum must be rejected")
report = dict(status="passed", upstream_commit=COMMIT, synthetic=True,
              clients=20, samples=len(p), rounds=3, calibration=False,
              official_methods_called=["obj_H", "local_fair_post", "true_H", "local_post_eval"],
              checks=["deterministic replay", "distinct client thresholds", "local constraint active",
                      "global constraint active", "unknown client rejected", "missing group rejected"],
              global_lambda=first.global_lambda,
              thresholds={str(c): t.tolist() for c,t in first.thresholds.items()})
print(json.dumps(report, indent=2))
