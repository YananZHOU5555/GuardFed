#!/usr/bin/env python3
"""Bounded validation-only FedAA screen; no test evaluation or automatic retries."""
import argparse
import copy
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time
import traceback


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def json_safe(value):
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


ADAPTATIONS = [
    "Official FedAA DDPG policy; common CNN initialization/local Adam/reweighting/project attacks",
    "Reward is raw accuracy on clean train-derived root only; no valid/test reward or calibration",
    "20 clients, keep10/16 as explicit search configuration, ascending client-ID tie order; full attacked models",
    "CPU policy and isolated RNG; actor/critic LR .001/.01 synchronized search; replay batch16/capacity100000; alpha0",
    "Virtual bootstrap enables the configured useful aggregations/policy updates; last transition remains pending",
    "All-zero distance guard; perfect-root terminal raises instead of changing upstream stopping policy",
    "Full train162770 / valid19867; 70-round validation search or explicit 3-round pipeline gate; never test",
]


def validate_job(job):
    cfg = job["config"]
    stage = job["evidence_stage"]
    if stage not in {"validation_screen", "pipeline_canary_only"}:
        raise ValueError("Unexpected evidence stage")
    expected = dict(seed=91001, num_clients=20, num_malicious=4, local_epochs=1,
                    batch_size=64, rounds=70 if stage == "validation_screen" else 3,
                    server_ratio=.1, synthetic_ratio=0., include_sensitive_feature=False,
                    optimizer="adam", full_round_diagnostics=True,
                    root_label_noise=0., root_sensitive_noise=0., celeba_train_limit=0,
                    celeba_eval_limit=0, celeba_evaluation_split="valid",
                    ad2_calibration_enabled=False)
    if any(cfg.get(k) != v for k, v in expected.items()):
        raise ValueError("Config differs from frozen screen/gate protocol")
    if (job["dataset"], job["method"]) != ("celeba", "FedAA-DDPG-adapted-v1"):
        raise ValueError("Unexpected method/data identity")
    if job["distribution"] not in {"IID", "non-IID"}:
        raise ValueError("Unexpected distribution")
    if cfg["client_alpha"] != {"IID":5000., "non-IID":5.}[job["distribution"]]:
        raise ValueError("Distribution/alpha mismatch")
    if job["attack"] not in ["Benign", "S-DFA"] or job["policy_seed"] != 91001:
        raise ValueError("Unexpected attack or policy seed")
    if cfg["learning_rate"] not in {.0005,.001} or job["aggre_num"] not in {10,16}:
        raise ValueError("Unexpected local LR or retention count")
    policy = job["policy_config"]
    if policy["actor_lr"] not in {.001,.01} or policy["critic_lr"] != policy["actor_lr"]:
        raise ValueError("Policy LR grid must remain synchronized")
    if set(policy) != {"actor_lr", "critic_lr"}:
        raise ValueError("Unfrozen policy configuration key")


def run(args):
    # Set before importing torch or creating any CUDA context.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import numpy as np
    import torch
    repo = Path(args.repo).resolve()
    sys.path.insert(0, str(repo / "scripts"))
    import reproduce_paper_tables as core
    from fedaa_round_adapter import FedAARounds, train_and_aggregate_round
    from fedaa_official_adapter import OFFICIAL_COMMIT, DDPG_SHA256

    torch.set_num_threads(int(os.environ.get("GUARDFED_CPU_THREADS", "1")))
    torch.set_num_interop_threads(1)
    job_path = Path(args.job).resolve()
    job = json.loads(job_path.read_text(encoding="utf-8"))
    validate_job(job)
    out = Path(args.out).resolve()
    if out.exists() and any(out.iterdir()) and not args.resume:
        raise FileExistsError("Nonempty output: do not rerun or overwrite a prior attempt")
    out.mkdir(parents=True, exist_ok=True)
    lock = out / "worker.lock"
    with lock.open("x") as f:
        f.write(json.dumps(dict(pid=os.getpid(), started_unix=time.time())))
    try:
        started = time.time()
        source_hashes = {}
        for name, expected in job["source_hashes"].items():
            source_hashes[name] = digest(repo / name)
            if source_hashes[name] != expected:
                raise ValueError(f"Frozen source/data changed: {name}")
        official = Path(args.official_dir).resolve()
        adapter_files = {
            "screen_runner": Path(__file__),
            "round_adapter": Path(__file__).with_name("fedaa_round_adapter.py"),
            "official_wrapper": Path(__file__).resolve().parents[2] / "fedaa/fedaa_official_adapter.py",
            "official_ddpg": official / "DDPG/DDPG.py",
        }
        adapter_hashes = {k: digest(v) for k, v in adapter_files.items()}
        if adapter_hashes != job["adapter_hashes"]:
            raise ValueError("Frozen screen adapter hashes changed")
        if adapter_hashes["official_ddpg"] != DDPG_SHA256:
            raise ValueError("Official DDPG hash mismatch")
        cfg = core.ExperimentConfig(**job["config"])
        device = core.choose_device(cfg.device)
        core.set_seed(cfg.seed, deterministic_image=True)
        bundle = core.load_bundle("celeba", cfg.client_alpha, cfg, device)
        contract = bundle["image_data_contract"]
        if (bundle["train_rows"], bundle["test_rows"], contract["evaluation_split"],
                contract["train_eval_disjoint"], contract["root_client_disjoint"],
                bundle["root_synthetic_rows"]) != (162770, 19867, "valid", True, True, 0):
            raise ValueError("Unexpected image data split or clean-root identity")
        if bundle["feature_includes_label"] or bundle["feature_includes_sensitive"]:
            raise ValueError("Image metadata leakage")
        environment = dict(python=sys.version, torch=torch.__version__, cuda=torch.version.cuda,
                           device=str(device), cpu_threads=torch.get_num_threads(),
                           gpu_name=torch.cuda.get_device_name(device) if device.type == "cuda" else None)
        identity = dict(job_sha256=digest(job_path), config=asdict(cfg), attack=job["attack"],
                        source_hashes=source_hashes, adapter_hashes=adapter_hashes,
                        data_contract=contract, environment=environment,
                        policy_seed=job["policy_seed"], policy_config=job["policy_config"],
                        aggre_num=job["aggre_num"], distribution=job["distribution"],
                        official_commit=OFFICIAL_COMMIT)
        record = dict(identity=identity, job=job, job_path=str(job_path), repo=str(repo),
                      output=str(out), adaptations=ADAPTATIONS, status="running",
                      evidence_stage=job["evidence_stage"], started_unix=started,
                      resume_from=str(Path(args.resume).resolve()) if args.resume else None)
        # Never overwrite an earlier record silently with a different protocol.
        record_path = out / "input_record.json"
        if record_path.exists() and json.loads(record_path.read_text(encoding="utf-8"))["identity"] != identity:
            raise ValueError("Output belongs to a different input identity")
        write_json(record_path, record)
        print(json.dumps(dict(job_id=job["id"], adaptations=ADAPTATIONS, identity=identity)), flush=True)
        core.set_seed(cfg.seed, deterministic_image=True)
        model = core.make_model(bundle, cfg, device)
        malicious = list(range(cfg.num_malicious))
        clients, audits = [], []
        for cid in range(cfg.num_clients):
            client, audit = core.client_runtime_data(bundle["clients"][cid], cid, job["attack"],
                malicious, bundle["rw_weights"], device, cfg.fflip_mode, cfg.foe_mode,
                cfg.sdfa_foe_mode, cfg.spdfa_foe_mode)
            clients.append(client)
            audits.append(audit)
        initial_root = core.evaluate_model(model, bundle["server_X"], bundle["server_y"],
                                          bundle["server_sensitive"], cfg.batch_size)["accuracy"]
        controller = FedAARounds(model, official, job["policy_seed"], initial_root,
                                participants=20, keep=job["aggre_num"], **job["policy_config"])
        trajectory, diagnostics, warnings = [], [], []
        if args.resume:
            saved = torch.load(args.resume, map_location="cpu", weights_only=False)
            if saved["identity"] != identity:
                raise ValueError("Checkpoint source/data/config/environment identity mismatch")
            if not (out / "checkpoint_round1.pt").exists():
                first_path = Path(args.resume).resolve().parent / "checkpoint_round1.pt"
                first_state = torch.load(first_path, map_location="cpu", weights_only=False)
                if first_state["identity"] != identity or first_state["controller"]["rounds"] != 1:
                    raise ValueError("Resume chain lacks matching round1 checkpoint")
                shutil.copyfile(first_path, out / "checkpoint_round1.pt")
            model.load_state_dict(saved["model"])
            controller.load_state_dict(saved["controller"])
            trajectory, diagnostics, audits, warnings = (saved[k] for k in
                ("trajectory_metrics", "round_summaries", "attack_audit", "warnings"))
            if controller.rounds != len(trajectory) or controller.rounds != len(diagnostics):
                raise ValueError("Incomplete checkpoint history")
            torch.set_rng_state(saved["rng"]["torch_cpu"])
            if device.type == "cuda":
                torch.cuda.set_rng_state_all(saved["rng"]["torch_cuda"])
            np.random.set_state(saved["rng"]["numpy"])
            random.setstate(saved["rng"]["python"])
        stop_after = args.stop_after or cfg.rounds
        if not controller.rounds < stop_after <= cfg.rounds:
            raise ValueError("Stop boundary must be after checkpoint and at most configured final round")
        for rnd in range(controller.rounds + 1, stop_after + 1):
            diag = train_and_aggregate_round(core, model, bundle, clients, audits, cfg, controller, device)
            attack_warnings = core.validate_attack_audit(job["attack"], malicious, audits)
            if any(w.startswith("client") or w.startswith("Sp-DFA") for w in attack_warnings):
                raise RuntimeError("Attack self-check failed: " + "; ".join(attack_warnings))
            warnings = sorted(set(warnings + attack_warnings))
            metrics = core.evaluate_model(model, bundle["X_test"], bundle["y_test"],
                                          bundle["test_sensitive"], cfg.batch_size)
            if metrics["prediction_count"] != 19867 or not all(math.isfinite(metrics[k]) for k in core.METRICS):
                raise ValueError("Nonfinite/incomplete official validation metrics")
            if controller.policy.transitions != rnd or len(diag["selected_client_ids"]) != job["aggre_num"]:
                raise ValueError("Policy transition/retention mismatch")
            if any(not torch.isfinite(v).all() for v in model.state_dict().values()):
                raise FloatingPointError("CNN contains nonfinite parameters")
            trajectory.append(dict(round=rnd, metrics={k: metrics[k] for k in core.METRICS},
                                   evaluation_stats={k: metrics[k] for k in
                                       ("positive_rate", "majority_accuracy", "prediction_count")}))
            diagnostics.append(dict(round=rnd, aggregate=diag, client_ids=list(range(20)),
                                    malicious_mask=[cid in malicious for cid in range(20)]))
            rng = dict(torch_cpu=torch.get_rng_state(), numpy=np.random.get_state(),
                       python=random.getstate(),
                       torch_cuda=torch.cuda.get_rng_state_all() if device.type == "cuda" else [])
            checkpoint = dict(identity=identity, model=copy.deepcopy(model.state_dict()),
                              controller=controller.state_dict(), rng=rng,
                              trajectory_metrics=trajectory, round_summaries=diagnostics,
                              attack_audit=audits, warnings=warnings)
            tmp = out / "training_state.pt.tmp"
            torch.save(checkpoint, tmp)
            tmp.replace(out / "training_state.pt")
            if rnd == 1:
                # Retain the exact first boundary for a separate recovery branch.
                first = out / "checkpoint_round1.pt"
                if first.exists():
                    raise FileExistsError("Refusing to replace retained round1 checkpoint")
                torch.save(checkpoint, first)
            progress = dict(job_id=job["id"], round=rnd, total_rounds=cfg.rounds, pid=os.getpid(),
                            metrics=trajectory[-1], policy_updates=controller.policy.transitions,
                            updated_unix=time.time(), elapsed_sec=time.time() - started)
            write_json(out / "progress.json", progress)
            print(json.dumps(progress), flush=True)
        model_tmp = out / "model.pt.tmp"
        torch.save(model.state_dict(), model_tmp)
        model_tmp.replace(out / "model.pt")
        status = ("screen_complete" if job["evidence_stage"] == "validation_screen" else "pilot_complete") if controller.rounds == cfg.rounds else "boundary_saved"
        result = dict(schema="fedaa_validation_screen_v1", id=job["id"], status=status, evidence_stage=job["evidence_stage"], method=job["method"],
                      dataset="celeba", distribution=job["distribution"], attack=job["attack"],
                      seed=cfg.seed, rounds=controller.rounds, planned_rounds=cfg.rounds,
                      config=asdict(cfg), policy_config=job["policy_config"], aggre_num=job["aggre_num"],
                      tuning_candidate=job["tuning_candidate"], metrics=trajectory[-1]["metrics"],
                      trajectory_metrics=trajectory, round_summaries=diagnostics,
                      attack_audit=audits, warnings=warnings, identity=identity,
                      source_hashes=source_hashes, adapter_hashes=adapter_hashes,
                      data_contract=contract,
                      adaptations=ADAPTATIONS, checkpoint_sha256=digest(out / "model.pt"),
                      full_checkpoint_sha256=digest(out / "training_state.pt"),
                      first_checkpoint_sha256=digest(out / "checkpoint_round1.pt"),
                      duration_sec=time.time() - started)
        write_json(out / "result.json", json_safe(result))
        record.update(status=status, finished_unix=time.time())
        write_json(record_path, record)
        print(json.dumps(dict(status=status, output=str(out), rounds=controller.rounds)), flush=True)
    except BaseException as exc:
        failure = dict(exception=type(exc).__name__, message=str(exc), traceback=traceback.format_exc(),
                       failed_unix=time.time(), evidence_stage=job["evidence_stage"])
        path = out / f"failure_{time.time_ns()}.json"
        write_json(path, failure)
        raise
    finally:
        lock.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--official-dir", default=str(Path(__file__).resolve().parents[2] / "fedaa/official"))
    parser.add_argument("--resume", help="Trusted full training_state.pt or checkpoint_round1.pt")
    parser.add_argument("--stop-after", type=int)
    run(parser.parse_args())
