import json
import math
import os
from pathlib import Path

import pandas as pd


RAW = Path("/home/yannan/workspace/GuardFed/results/paper_tables/raw_results.jsonl")
OUT = Path("/home/yannan/workspace/GuardFed/results/paper_tables/expanded_experiments")
SUITE = os.environ.get("SYNTH_SUITE", "synthetic_clean_cap10_dense_v3")


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def score(acc, aeod, aspd):
    return float(acc) - 0.5 * (float(aeod) + float(aspd))


def select_joint_last10(record):
    best = None
    for item in record.get("last10_metrics") or []:
        m = item.get("metrics", {})
        vals = (m.get("accuracy"), m.get("aeod"), m.get("aspd"))
        if not all(finite(v) for v in vals):
            continue
        cand = {
            "selected_round": item.get("round"),
            "acc": float(vals[0]),
            "aeod": float(vals[1]),
            "aspd": float(vals[2]),
            "score": score(*vals),
            "selection": "best_joint_last10",
        }
        if best is None or cand["score"] > best["score"]:
            best = cand
    if best is not None:
        return best
    m = record["metrics"]
    vals = (m.get("accuracy"), m.get("aeod"), m.get("aspd"))
    return {
        "selected_round": record.get("rounds"),
        "acc": float(vals[0]),
        "aeod": float(vals[1]),
        "aspd": float(vals[2]),
        "score": score(*vals),
        "selection": "final_round_fallback",
    }


def label(cfg):
    return f"{int(round(float(cfg.get('server_ratio') or 0.0) * 100))}% real clean"


def rank_columns(frame, groups):
    out = frame.copy()
    out["score_rank"] = out.groupby(groups)["score_mean"].rank(method="min", ascending=False)
    out["acc_rank"] = out.groupby(groups)["acc_mean"].rank(method="min", ascending=False)
    out["aeod_rank"] = out.groupby(groups)["aeod_mean"].rank(method="min", ascending=True)
    out["aspd_rank"] = out.groupby(groups)["aspd_mean"].rank(method="min", ascending=True)
    return out


def main():
    rows = []
    with RAW.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            cfg = r.get("config", {})
            if cfg.get("experiment_suite") != SUITE:
                continue
            rows.append(
                {
                    "dataset": r.get("dataset"),
                    "distribution": r.get("distribution"),
                    "attack": r.get("attack"),
                    "method": r.get("method"),
                    "seed": cfg.get("seed"),
                    "setting": label(cfg),
                    "server_ratio": float(cfg.get("server_ratio") or 0.0),
                    "server_sampling": cfg.get("server_sampling"),
                    "tag": cfg.get("experiment_tag"),
                    **select_joint_last10(r),
                }
            )
    if not rows:
        raise SystemExit(f"No rows for {SUITE}")
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    raw_csv = OUT / f"{SUITE}_joint_raw.csv"
    by_dataset_csv = OUT / f"{SUITE}_by_dataset.csv"
    by_slice_csv = OUT / f"{SUITE}_by_slice.csv"
    md = OUT / f"{SUITE}_summary.md"
    df.to_csv(raw_csv, index=False)
    agg = dict(
        n=("score", "size"),
        acc_mean=("acc", "mean"),
        acc_std=("acc", "std"),
        aeod_mean=("aeod", "mean"),
        aeod_std=("aeod", "std"),
        aspd_mean=("aspd", "mean"),
        aspd_std=("aspd", "std"),
        score_mean=("score", "mean"),
        score_std=("score", "std"),
        round_mean=("selected_round", "mean"),
    )
    by_dataset = (
        df.groupby(["dataset", "setting", "server_ratio", "server_sampling"], dropna=False)
        .agg(**agg)
        .reset_index()
    )
    by_dataset = rank_columns(by_dataset, ["dataset"]).sort_values(["dataset", "score_rank"])
    by_dataset.to_csv(by_dataset_csv, index=False)
    by_slice = (
        df.groupby(["dataset", "distribution", "attack", "setting", "server_ratio", "server_sampling"], dropna=False)
        .agg(**agg)
        .reset_index()
    )
    by_slice = rank_columns(by_slice, ["dataset", "distribution", "attack"]).sort_values(
        ["dataset", "distribution", "attack", "score_rank"]
    )
    by_slice.to_csv(by_slice_csv, index=False)

    lines = [
        f"# Synthetic clean-only cap10 dense: {SUITE}",
        "",
        "10 ratios: 1%, 2%, 3%, 4%, 5%, 6%, 7%, 8%, 9%, 10% clean server/root data; no synthetic rows.",
        "Each setting averages 3 seeds, 2 distributions, and 2 attacks at dataset level.",
        "Selection per run: best joint checkpoint in the last 10 rounds, `ACC - 0.5*(AEOD+ASPD)`.",
        "",
        "## Dataset-level ranking",
        "",
        by_dataset.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## 10% real clean check",
        "",
        by_dataset[by_dataset["setting"] == "10% real clean"].to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Slice-level 10% rows",
        "",
        by_slice[by_slice["setting"] == "10% real clean"].to_markdown(index=False, floatfmt=".4f"),
        "",
    ]
    md.write_text("\n".join(lines), encoding="utf-8")
    print(f"rows={len(df)}")
    print(f"wrote={raw_csv}")
    print(f"wrote={by_dataset_csv}")
    print(f"wrote={by_slice_csv}")
    print(f"wrote={md}")
    print(by_dataset[["dataset", "setting", "n", "acc_mean", "aeod_mean", "aspd_mean", "score_mean", "score_rank", "acc_rank", "aeod_rank", "aspd_rank"]].to_string(index=False))


if __name__ == "__main__":
    main()
