"""Trace submitted Table II against retained records; never retrain or alter sources."""
from pathlib import Path
from collections import defaultdict, Counter
import hashlib
import json
import math
import statistics

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
METRICS = {"ACC": "accuracy", "AEOD": "aeod", "ASPD": "aspd"}
INPUTS = [
    ROOT / "tmp/table2_rawaudit_20261002/paper_tables_raw.jsonl",
    ROOT / "tmp/table2_rawaudit_20261002/attack_strength_raw.jsonl",
    ROOT / "outputs/guardfed_tables/raw_results_5090_clean10_full.jsonl",
]
CELLS = ROOT / "tmp/table2_trace_20261002/submission_table2_cells.json"
MAP = ROOT / "tmp/table2_trace_20261002/submission_table2_map.json"
ALIASES = {
    "FedWA": ["AdaAggRL", "FedAMM"],
    "FairGuard → FLTrust": ["FLTrust+FairGuard"],
    "Cosine Similarity + Fairness Deviation": ["GuardFed"],
}


def digest(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def compatible(display, value, metric):
    scale = 100 if metric == "ACC" else 1
    digits = len(display.split(".")[1]) if "." in display else 0
    return math.isfinite(value) and abs(float(display) - value * scale) <= 0.5 * 10 ** -digits + 1e-12


def projections(r):
    yield "final_round70_same_checkpoint", r["metrics"], {k: [70] for k in METRICS.values()}
    history = r.get("last10_metrics", [])
    if len(history) == 10 and [x["round"] for x in history] == list(range(61, 71)):
        v = {k: (max if k == "accuracy" else min)(x["metrics"][k] for x in history) for k in METRICS.values()}
        rounds = {k: [x["round"] for x in history if x["metrics"][k] == v[k]] for k in v}
        yield "legacy_per_metric_last10_extrema_not_one_checkpoint", v, rounds
        yield "last10_mean_not_final_checkpoint", {k: statistics.mean(x["metrics"][k] for x in history) for k in v}, {k: list(range(61, 71)) for k in v}


def main():
    cells = json.loads(CELLS.read_text(encoding="utf-8"))
    meta = json.loads(MAP.read_text(encoding="utf-8"))
    assert len(cells) == 480
    conditions = defaultdict(dict)
    for c in cells:
        conditions[(c["method"], c["distribution"], c["attack"])][c["metric"]] = c
    index = defaultdict(list)
    source_hashes = []
    for p in INPUTS:
        source_hashes.append({"path": str(p.relative_to(ROOT)), "sha256": digest(p), "bytes": p.stat().st_size})
        for line_no, line in enumerate(p.open(encoding="utf-8"), 1):
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("dataset") != "adult" or r.get("rounds") != 70 or r.get("mode") == "smoke" or r.get("study") == "ratio":
                continue
            index[(r["method"], r["distribution"], r["attack"])].append((r, str(p.relative_to(ROOT)), line_no))
    traced = []
    hidden = []
    for (method, distribution, attack), published in conditions.items():
        pool = []
        for backend in ALIASES.get(method, [method]):
            pool.extend(index[(backend, distribution, "F Flip" if attack == "F-Flip" else attack)])
        candidates = []
        unique = set()
        for r, source, line_no in pool:
            for rule, values, rounds in projections(r):
                if not all(c["numeric_value"] is None or compatible(c["display"], values[METRICS[m]], m) for m, c in published.items()):
                    continue
                identity = json.dumps([r["method"], r["seed"], r["config"], rule, values, rounds], sort_keys=True)
                if identity in unique:
                    continue
                unique.add(identity)
                candidates.append({"raw_method": r["method"], "seed": r["seed"], "run_id": r["run_id"], "source": source, "line": line_no, "config": r["config"], "config_sha256": hashlib.sha256(json.dumps(r["config"], sort_keys=True).encode()).hexdigest(), "rule": rule, "metrics": values, "metric_rounds": rounds, "final_metrics": r["metrics"], "evaluation_stats": r.get("evaluation_stats"), "data_contract": r.get("data_contract")})
        record = {"method": method, "distribution": distribution, "attack": attack, "published": {m: c["display"] for m, c in published.items()}, "name_mapping": "unverified_candidate_alias" if method in ALIASES else "same_record_name", "numeric_compatible_candidates": candidates, "candidate_count": len(candidates), "status": "NUMERIC_COMPATIBILITY_ONLY" if candidates else "NO_MATCH_IN_CHECKED_PROJECTIONS"}
        traced.append(record)
        for m, c in published.items():
            if c["numeric_value"] is not None:
                continue
            vals = sorted({x["metrics"][METRICS[m]] for x in candidates})
            hidden.append({"method": method, "distribution": distribution, "attack": attack, "metric": m, "published": c["display"], "candidate_count": len(candidates), "candidate_values": vals, "status": "CANDIDATE_VALUE_AGREEMENT_NOT_PROVEN_SOURCE" if len(vals) == 1 else "AMBIGUOUS_CANDIDATE_VALUES" if vals else "NO_MATCHED_RECORD"})
    assert len(traced) == 160 and len(hidden) == 44
    summary = {"status": "PARTIAL_PROVENANCE_AUDIT_NOT_FINAL_TABLE", "submission_pdf_sha256": meta["submission_pdf_sha256"], "dataset": "adult", "conditions": 160, "cells": 480, "numeric_cells": 436, "hidden_cells": 44, "condition_status": dict(Counter(x["status"] for x in traced)), "hidden_status": dict(Counter(x["status"] for x in hidden)), "source_files": source_hashes, "rules_checked": ["round70", "last10_per_metric_extrema", "last10_mean"], "source_identity_limit": "Numeric agreement, even all visible metrics, does not prove generating source or exact source-code/data/checkpoint identity. Unmatched records are not declared lost. FOE is never used as a substitute for FedSA.", "new_training": 0}
    for name, value in [("trace_summary.json", summary), ("condition_candidates.json", traced), ("hidden_value_candidates.json", hidden), ("submission_table2_cells.json", cells), ("submission_table2_map.json", meta)]:
        (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# Table II：隐藏值的候选追溯", "", "这是来源审计，不是可直接替换的论文表。数字匹配只表示兼容，不能证明原表生成来源。候选名映射未认证；不附加其他配置的标准差。", "", "| 方法 | 分布 | 场景 | 指标 | 候选记录数 | 候选值范围 | 状态 |", "|---|---|---|---|---:|---|---|"]
    for h in hidden:
        v = h["candidate_values"]
        display = f"{v[0]:.6f}" if len(v) == 1 else f"{v[0]:.6f}–{v[-1]:.6f}" if v else "尚未找到匹配"
        lines.append(f"| {h['method']} | {h['distribution']} | {h['attack']} | {h['metric']} | {h['candidate_count']} | {display} | {h['status']} |")
    (OUT / "隐藏值候选清单.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ["conditions", "cells", "hidden_cells", "condition_status", "hidden_status", "new_training"]}, ensure_ascii=False))


if __name__ == "__main__":
    assert compatible("83.00", 0.83, "ACC")
    assert not compatible("83.00", 0.831, "ACC")
    assert compatible("0.0559", 0.05591, "AEOD")
    main()
