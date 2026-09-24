"""Score-only LoGoFair DP prototype calling pinned official optimizer methods.

This is not the old GuardFed two-global-threshold substitute. Each client retains
its own local multiplier; a global multiplier is averaged every post round.
Validation-only group priors deliberately replace upstream's inconsistent
train+valid+test counts / valid denominator. See AUDIT.md before integration.
"""
import ast
import contextlib
import hashlib
import io
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

COMMIT = "7044815cf813cdad53fca7bab426c4d2194ab505"
SOURCE_SHA256 = "244e8bd3f3a6f5b4945cd8cf93035d5996954acc7f5b61e41e89651d9f88fba2"


def _official_client(source, calibration):
    raw = Path(source).read_bytes().replace(b"\r\n", b"\n")
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
        raise ValueError("Official client source differs from the audited snapshot")
    tree = ast.parse(raw.decode("utf-8"))
    cls = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == "FFPClient")
    names = {"r_beta", "obj_H", "true_H", "local_fair_post", "local_post_eval", "calibration"}
    # Avoid importing unused model training/cvxpy dependencies. Method ASTs are
    # executed verbatim; EO is explicitly unsupported, not silently approximated.
    cls.body = [x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name in names]
    namespace = {"torch": torch, "np": np}
    if calibration:
        from netcal.scaling import BetaCalibration
        namespace["BetaCalibration"] = BetaCalibration
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["FFPClient"]


def _inputs(probability, sensitive, client_id):
    p = torch.as_tensor(probability, dtype=torch.float32).detach().cpu().clone().reshape(-1)
    a = torch.as_tensor(sensitive).detach().cpu().clone().reshape(-1)
    cid = np.asarray(client_id).reshape(-1)
    if not len(p) or len(p) != len(a) or len(p) != len(cid):
        raise ValueError("Nonempty equal-length probability, sensitive, client_id required")
    if not torch.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Scores must be finite P(Y=1), not logits or margins")
    if not torch.isin(a, torch.tensor([0, 1])).all():
        raise ValueError("Sensitive groups must be 0/1")
    if not np.issubdtype(cid.dtype, np.integer):
        raise ValueError("Client IDs must be stable integers")
    return p, a, cid


class OfficialLoGoFairDP:
    def __init__(self, source=None, *, global_delta=.02, local_delta=.02,
                 post_rounds=30, local_steps=20, global_steps=20, post_lr=.005,
                 beta=1000., calibration=True):
        self.source = Path(source) if source else Path(__file__).parent / "upstream/fedlearn/models/FedFairPostClient.py"
        for name, value in [("global_delta", global_delta), ("local_delta", local_delta), ("post_lr", post_lr), ("beta", beta)]:
            if not np.isfinite(value) or value < 0 or (name in {"beta", "post_lr"} and value == 0):
                raise ValueError(name)
        if any(not isinstance(x, int) or x < 1 for x in [post_rounds, local_steps, global_steps]):
            raise ValueError("Positive integer post/local/global rounds required")
        self.settings = dict(global_delta=global_delta, local_delta=local_delta,
                             post_rounds=post_rounds, local_steps=local_steps,
                             global_steps=global_steps, post_lr=post_lr, beta=beta,
                             calibration=calibration)

    def fit(self, probability, label, sensitive, client_id):
        p, a, cid = _inputs(probability, sensitive, client_id)
        y = torch.as_tensor(label).detach().cpu().clone().reshape(-1)
        if len(y) != len(p) or not torch.isin(y, torch.tensor([0, 1])).all():
            raise ValueError("Equal-length binary validation labels required")
        ids = np.unique(cid)
        official = _official_client(self.source, self.settings["calibration"])
        self.clients, self.thresholds = {}, {}
        group_counts = [int((a == g).sum()) for g in [0, 1]]
        counts = {int(c): {g: int(((a == g) & torch.from_numpy(cid == c)).sum()) for g in [0, 1]} for c in ids}
        for c in ids:
            if min(counts[int(c)].values()) == 0:
                raise ValueError(f"Client {c} has an absent sensitive group; cannot apply official objective")
            mask = torch.from_numpy(cid == c)
            obj = official()
            obj.cid, obj.client_num = int(c), len(ids)
            obj.val_data = SimpleNamespace(A=a[mask], Y=y[mask])
            obj.val_score = p[mask].clone()
            obj.data_info = {"val_client_A_info": counts, "val_num": len(p)}
            obj.N_0_c, obj.N_1_c = [torch.tensor(counts[int(c)][g]) for g in [0, 1]]
            obj.pi_0_c, obj.pi_1_c = [torch.tensor(counts[int(c)][g] / len(p)) for g in [0, 1]]
            obj.p_A_0, obj.p_A_1 = [torch.tensor(group_counts[g] / len(p)) for g in [0, 1]]
            obj.fair_metric = "DP"
            obj.fairness_constraints = {"fairness_measure": "DP"}
            obj.global_delta = torch.tensor(self.settings["global_delta"])
            obj.local_delta = torch.tensor(self.settings["local_delta"])
            obj.FFP_beta, obj.post_lr = self.settings["beta"], self.settings["post_lr"]
            obj.post_local_round_mu, obj.post_local_round_lamb = self.settings["local_steps"], self.settings["global_steps"]
            obj.local_mu = torch.tensor([.5, .5])
            obj.calib = self.settings["calibration"]
            if obj.calib:
                for g in [0, 1]:
                    if len(torch.unique(obj.val_data.Y[obj.val_data.A == g])) < 2:
                        raise ValueError(f"Client {c} group {g} lacks both labels for beta calibration")
                obj.calibration(train=True)
            obj.val_dataloader = obj.test_dataloader = None
            # Capture exact upstream final thresholds without evaluating test data.
            def capture(*args, _cid=int(c), post_threshold=None, **kwargs):
                self.thresholds[_cid] = torch.cat(post_threshold).detach().clone()
                return {}
            obj.model_eval = capture
            self.clients[int(c)] = obj
        lamb = torch.tensor([.01])  # official server initialization
        self.history = []
        with contextlib.redirect_stdout(io.StringIO()):
            for round_id in range(self.settings["post_rounds"]):
                updates = []
                for obj in self.clients.values():
                    obj.global_lamb = lamb
                    updates.append(obj.local_fair_post())
                lamb = (sum(updates) / len(ids)).detach()
                if not torch.isfinite(lamb).all():
                    raise FloatingPointError("Official global update diverged; no silent retry or clamp")
                for obj in self.clients.values():
                    obj.global_lamb = lamb
                    obj.local_post_eval()  # includes official true_H local refinement
                    if not torch.isfinite(self.thresholds[obj.cid]).all():
                        raise FloatingPointError("Official local threshold diverged")
                self.history.append({"round": round_id + 1, "global_lambda": float(lamb.item())})
        self.global_lambda = float(lamb.item())
        return self

    def predict(self, probability, sensitive, client_id):
        p, a, cid = _inputs(probability, sensitive, client_id)
        if set(np.unique(cid)) - set(self.clients):
            raise ValueError("Evaluation client ID missing from fitted validation clients")
        result = torch.empty_like(p)
        for c, obj in self.clients.items():
            mask = torch.from_numpy(cid == c)
            if not mask.any():
                continue
            score, group = p[mask].clone(), a[mask]
            if obj.calib:
                # netcal rejects empty arrays, so transform only present groups.
                for g, calibrator in [(0, obj.score_calibrated_0), (1, obj.score_calibrated_1)]:
                    gm = group == g
                    if gm.any():
                        score[gm] = torch.as_tensor(calibrator.transform(score[gm].numpy()), dtype=torch.float32)
            threshold = self.thresholds[c][group.long()]
            # Upstream returns 0.5 on exact ties, incompatible with binary metrics.
            # Fail explicitly rather than silently resolve them differently.
            if (score == threshold).any():
                raise ValueError("Exact score/threshold tie: freeze a binary tie policy before evaluation")
            result[mask] = (score > threshold).float()
        return result.numpy().astype(int)
