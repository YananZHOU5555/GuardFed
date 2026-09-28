"""Freeze 32 validation jobs or two default-parameter regression gates; never launch."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

from run_fedaa_screen import ADAPTATIONS, digest, validate_job

HERE = Path(__file__).resolve().parent
BASELINES = HERE.parents[1]
SCORE = "ACC-.35*(.45*AEOD+.45*ASPD+.10*max(AEOD,ASPD))-.10*max(0,max(AEOD,ASPD)-.06)"


def prepare(base_path, out, gates=False):
    base = json.loads(base_path.read_text(encoding="utf-8"))
    if base["dataset"] != "celeba" or base["distribution"] != "non-IID" or base["config"]["seed"] != 91001:
        raise ValueError("Expected immutable Stage A non-IID seed91001 config")
    stage = "pipeline_canary_only" if gates else "validation_screen"
    rounds = 3 if gates else 70
    out.mkdir(parents=True, exist_ok=True)
    (out / "jobs").mkdir(exist_ok=True)
    hashes = {"screen_runner": digest(HERE / "run_fedaa_screen.py"),
              "round_adapter": digest(HERE / "fedaa_round_adapter.py"),
              "official_wrapper": digest(BASELINES / "fedaa/fedaa_official_adapter.py"),
              "official_ddpg": digest(BASELINES / "fedaa/official/DDPG/DDPG.py")}
    grid = [(.01, 10, .001, "non-IID")] if gates else list(itertools.product(
        [.001, .01], [10, 16], [.0005, .001], ["IID", "non-IID"]))
    jobs = []
    for (policy_lr, keep, local_lr, distribution), attack in itertools.product(grid, ["Benign", "S-DFA"]):
        candidate = f"policy{policy_lr:g}_keep{keep}_local{local_lr:g}"
        ident = f"FedAA-DDPG_{candidate}_{distribution}_{attack}_seed91001"
        if gates:
            ident += "_gate3"
        cfg = dict(base["config"], learning_rate=local_lr, rounds=rounds,
                   client_alpha={"IID":5000., "non-IID":5.}[distribution],
                   ad2_calibration_enabled=False, full_round_diagnostics=True,
                   experiment_suite="fedaa_validation_screen_v1" if not gates else "fedaa_screen_regression_gate_v1",
                   experiment_tag=ident)
        job = dict(id=ident, dataset="celeba", distribution=distribution,
                   method="FedAA-DDPG-adapted-v1", attack=attack, config=cfg,
                   policy_seed=91001, aggre_num=keep,
                   policy_config=dict(actor_lr=policy_lr, critic_lr=policy_lr),
                   tuning_candidate=candidate, evidence_stage=stage,
                   source_hashes=base["source_hashes"], adapter_hashes=hashes,
                   adaptations=ADAPTATIONS,
                   derivation=dict(base_job_id=base["id"], base_job_sha256=digest(base_path),
                     base_source_output=base["output"],
                     changes={k: dict(before=base["config"].get(k), after=v)
                              for k, v in cfg.items() if base["config"].get(k) != v},
                     attack_override=dict(before=base["attack"], after=attack)))
        validate_job(job)
        path = out / "jobs" / f"{ident}.json"
        text = json.dumps(job, indent=2, ensure_ascii=False)
        if path.exists() and path.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"Frozen job changed: {path}")
        path.write_text(text, encoding="utf-8")
        jobs.append(dict(id=ident, path=str(path.relative_to(out)), sha256=digest(path),
                         tuning_candidate=candidate, distribution=distribution, attack=attack))
    expected = 2 if gates else 32
    assert len(jobs) == expected and len({j["id"] for j in jobs}) == expected
    manifest = dict(protocol="fedaa_validation_screen_v1", evidence_stage=stage,
                    new_run_count=expected, rounds=rounds, seeds=[91001], jobs=jobs,
                    candidate_count=1 if gates else 8, adapter_hashes=hashes,
                    source_hashes=base["source_hashes"], score=SCORE,
                    selection="Freeze existing score; report all candidates, raw ACC/AEOD/ASPD, accuracy champions and Pareto sets. n=1; no significance claim; no test-based selection.",
                    stopping="One attempt by default; failures retained. Manual exact-identity resume only after diagnosis. No automatic new configurations.",
                    old_results_rerun=False, training_started=False)
    path = out / "manifest.json"
    text = json.dumps(manifest, indent=2, ensure_ascii=False)
    if path.exists() and path.read_text(encoding="utf-8") != text:
        raise FileExistsError("Manifest differs from existing freeze")
    path.write_text(text, encoding="utf-8")
    print(json.dumps(dict(manifest=str(path), jobs=expected, rounds=rounds, training_started=False)))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-job", type=Path, default=BASELINES / "integration_20260928/base_job.json")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--pipeline-gates", action="store_true")
    args = parser.parse_args()
    prepare(args.base_job, args.out or HERE / ("gate_jobs" if args.pipeline_gates else "screen_jobs"), args.pipeline_gates)
