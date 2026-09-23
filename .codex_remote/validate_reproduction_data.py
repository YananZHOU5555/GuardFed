#!/usr/bin/env python3
"""Validate GuardFed Table II/III reproduction data and metrics.

This script is intentionally conservative: it checks the copied raw data,
sensitive attribute mappings, feature/label separation, and fairness metric
math before any full reproduction run is allowed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results" / "paper_tables"

ADULT_COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "income",
]

REQUIRED_FILES = {
    "adult": ["adult.data", "adult.test", "adult.names"],
    "compas": ["compas-scores-two-years.csv"],
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_manifest(dataset: str) -> List[Dict[str, Any]]:
    folder = DATA_DIR / dataset
    out = []
    for name in REQUIRED_FILES[dataset]:
        p = folder / name
        out.append({
            "dataset": dataset,
            "file": str(p),
            "exists": p.exists(),
            "size": p.stat().st_size if p.exists() else None,
            "sha256": sha256_file(p) if p.exists() else None,
        })
    return out


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, sensitive: np.ndarray) -> Dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    sensitive = np.asarray(sensitive).astype(int)
    if not (len(y_true) == len(y_pred) == len(sensitive)):
        raise ValueError("y_true, y_pred, and sensitive must have the same length")

    warnings: List[str] = []
    accuracy = float(np.mean(y_true == y_pred)) if len(y_true) else math.nan

    tprs = {}
    pred_rates = {}
    for group in [0, 1]:
        group_mask = sensitive == group
        positive_mask = group_mask & (y_true == 1)
        positive_count = int(positive_mask.sum())
        group_count = int(group_mask.sum())
        if positive_count == 0:
            warnings.append(f"AEOD denominator is zero for sensitive group {group}")
            tprs[group] = math.nan
        else:
            tprs[group] = float(np.mean(y_pred[positive_mask] == 1))
        if group_count == 0:
            warnings.append(f"ASPD denominator is zero for sensitive group {group}")
            pred_rates[group] = math.nan
        else:
            pred_rates[group] = float(np.mean(y_pred[group_mask] == 1))

    aeod = abs(tprs[0] - tprs[1]) if not (math.isnan(tprs[0]) or math.isnan(tprs[1])) else math.nan
    aspd = abs(pred_rates[0] - pred_rates[1]) if not (math.isnan(pred_rates[0]) or math.isnan(pred_rates[1])) else math.nan
    return {
        "accuracy": accuracy,
        "aeod": float(aeod) if not math.isnan(aeod) else math.nan,
        "aspd": float(aspd) if not math.isnan(aspd) else math.nan,
        "tpr_group0": tprs[0],
        "tpr_group1": tprs[1],
        "pred_rate_group0": pred_rates[0],
        "pred_rate_group1": pred_rates[1],
        "warnings": warnings,
    }


def metric_self_check(name: str) -> Dict[str, Any]:
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    sensitive = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    y_pred = np.array([1, 0, 1, 1, 1, 0, 0, 0])
    metrics = compute_metrics(y_true, y_pred, sensitive)
    expected = {"accuracy": 0.75, "aeod": 0.5, "aspd": 0.0}
    for key, value in expected.items():
        if not math.isclose(metrics[key], value, rel_tol=1e-12, abs_tol=1e-12):
            raise AssertionError(f"{name} metric self-check failed for {key}: {metrics[key]} != {value}")

    zero_case = compute_metrics(
        np.array([0, 0, 0, 1]),
        np.array([0, 1, 0, 1]),
        np.array([0, 0, 1, 1]),
    )
    if not any("AEOD denominator is zero" in w for w in zero_case["warnings"]):
        raise AssertionError(f"{name} zero-denominator AEOD warning was not emitted")
    return {"name": name, "metrics": metrics, "zero_denominator_check": zero_case}


def load_adult_raw() -> Tuple[pd.DataFrame, pd.DataFrame]:
    folder = DATA_DIR / "adult"
    train_path = folder / "adult.data"
    test_path = folder / "adult.test"
    train_df = pd.read_csv(
        train_path,
        names=ADULT_COLUMNS,
        sep=r"\s*,\s*",
        engine="python",
        na_values="?",
        skiprows=0,
    )
    test_df = pd.read_csv(
        test_path,
        names=ADULT_COLUMNS,
        sep=r"\s*,\s*",
        engine="python",
        na_values="?",
        skiprows=1,
    )
    for df in [train_df, test_df]:
        string_cols = [
            col for col in df.columns
            if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col])
        ]
        for col in string_cols:
            # Preserve true NA values. With pandas 2.x, astype(str) converts
            # np.nan into the literal string "nan", which prevents dropna()
            # from removing Adult rows containing "?" markers.
            df[col] = df[col].map(lambda x: x.strip() if isinstance(x, str) else x)
    test_df["income"] = test_df["income"].str.rstrip(".")
    train_df = train_df.dropna().reset_index(drop=True)
    test_df = test_df.dropna().reset_index(drop=True)
    train_df["income"] = (train_df["income"] == ">50K").astype(int)
    test_df["income"] = (test_df["income"] == ">50K").astype(int)
    train_df["sex"] = (train_df["sex"] == "Male").astype(int)
    test_df["sex"] = (test_df["sex"] == "Male").astype(int)
    return train_df, test_df


def summarize_frame(df: pd.DataFrame, label_col: str, sensitive_col: str) -> Dict[str, Any]:
    label_counts = {str(k): int(v) for k, v in df[label_col].value_counts().sort_index().items()}
    sensitive_counts = {str(k): int(v) for k, v in df[sensitive_col].value_counts().sort_index().items()}
    return {
        "rows": int(len(df)),
        "label_counts": label_counts,
        "label_positive_rate": float(df[label_col].mean()),
        "sensitive_counts": sensitive_counts,
        "sensitive_privileged_rate": float(df[sensitive_col].mean()),
    }


def validate_adult() -> Dict[str, Any]:
    train_df, test_df = load_adult_raw()
    errors = []
    if not 30150 <= len(train_df) <= 30170:
        errors.append(f"Adult train row count after dropna is {len(train_df)}, expected about 30162")
    if not 15050 <= len(test_df) <= 15070:
        errors.append(f"Adult test row count after dropna is {len(test_df)}, expected about 15060")
    if sorted(train_df["income"].unique().tolist()) != [0, 1]:
        errors.append("Adult income is not binary after mapping")
    if sorted(train_df["sex"].unique().tolist()) != [0, 1]:
        errors.append("Adult sex is not binary after mapping")
    feature_columns = [c for c in train_df.columns if c not in ["income", "sex"]]
    return {
        "dataset": "adult",
        "label": "income (>50K=1, <=50K=0)",
        "sensitive_attribute": "sex (Male=1, Female=0)",
        "feature_includes_label": False,
        "feature_includes_sensitive": False,
        "feature_columns": feature_columns,
        "num_features": len(feature_columns),
        "train": summarize_frame(train_df, "income", "sex"),
        "test": summarize_frame(test_df, "income", "sex"),
        "errors": errors,
    }


def load_compas_clean(seed: int = 123) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    path = DATA_DIR / "compas" / "compas-scores-two-years.csv"
    df = pd.read_csv(path)
    cleaned = df[
        (df["days_b_screening_arrest"] <= 30)
        & (df["days_b_screening_arrest"] >= -30)
        & (df["is_recid"] != -1)
        & (df["c_charge_degree"] != "O")
        & (df["score_text"] != "N/A")
    ].copy()
    features = [
        "sex", "age", "age_cat", "race",
        "juv_fel_count", "juv_misd_count", "juv_other_count",
        "priors_count", "c_charge_degree", "two_year_recid",
    ]
    cleaned = cleaned[features].dropna().reset_index(drop=True)
    cleaned["race"] = (cleaned["race"] == "African-American").astype(int)
    cleaned["sex"] = (cleaned["sex"] == "Male").astype(int)
    train_df, test_df = train_test_split(
        cleaned,
        test_size=0.3,
        random_state=seed,
        stratify=cleaned["race"],
    )
    return cleaned, train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def validate_compas(seed: int = 123) -> Dict[str, Any]:
    clean_df, train_df, test_df = load_compas_clean(seed=seed)
    errors = []
    if len(clean_df) < 6000:
        errors.append(f"COMPAS cleaned row count is {len(clean_df)}, unexpectedly low for ProPublica filter")
    if sorted(clean_df["two_year_recid"].unique().tolist()) != [0, 1]:
        errors.append("COMPAS two_year_recid is not binary")
    if sorted(clean_df["race"].unique().tolist()) != [0, 1]:
        errors.append("COMPAS race sensitive attribute is not binary after mapping")
    feature_columns = [c for c in clean_df.columns if c not in ["two_year_recid", "race"]]
    return {
        "dataset": "compas",
        "label": "two_year_recid",
        "sensitive_attribute": "race (African-American=1, Others=0)",
        "feature_includes_label": False,
        "feature_includes_sensitive": False,
        "feature_columns": feature_columns,
        "num_features": len(feature_columns),
        "cleaned_full": summarize_frame(clean_df, "two_year_recid", "race"),
        "train": summarize_frame(train_df, "two_year_recid", "race"),
        "test": summarize_frame(test_df, "two_year_recid", "race"),
        "cleaning_filter": "days_b_screening_arrest in [-30,30], is_recid!=-1, c_charge_degree!='O', score_text!='N/A'",
        "errors": errors,
    }


def write_reports(report: Dict[str, Any]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / "data_validation_report.json"
    md_path = RESULTS_DIR / "data_validation_report.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = []
    lines.append("# GuardFed Table II/III Data Validation Report")
    lines.append("")
    lines.append(f"Generated by `{Path(__file__).name}`.")
    lines.append("")
    lines.append("## Gate 1 Files")
    for item in report["files"]:
        status = "OK" if item["exists"] else "MISSING"
        lines.append(f"- {status}: `{item['file']}` size={item['size']} sha256={item['sha256']}")
    lines.append("")
    lines.append("## Adult")
    adult = report["adult"]
    lines.append(f"- Label: {adult['label']}")
    lines.append(f"- Sensitive: {adult['sensitive_attribute']}")
    lines.append(f"- Features: {adult['num_features']}; includes label={adult['feature_includes_label']}; includes sensitive={adult['feature_includes_sensitive']}")
    lines.append(f"- Train: {adult['train']}")
    lines.append(f"- Test: {adult['test']}")
    lines.append("")
    lines.append("## COMPAS")
    compas = report["compas"]
    lines.append(f"- Label: {compas['label']}")
    lines.append(f"- Sensitive: {compas['sensitive_attribute']}")
    lines.append(f"- Cleaning: {compas['cleaning_filter']}")
    lines.append(f"- Features: {compas['num_features']}; includes label={compas['feature_includes_label']}; includes sensitive={compas['feature_includes_sensitive']}")
    lines.append(f"- Cleaned full: {compas['cleaned_full']}")
    lines.append(f"- Train: {compas['train']}")
    lines.append(f"- Test: {compas['test']}")
    lines.append("")
    lines.append("## Metric Self-Checks")
    for item in report["metric_self_checks"]:
        m = item["metrics"]
        lines.append(f"- {item['name']}: ACC={m['accuracy']:.6f}, AEOD={m['aeod']:.6f}, ASPD={m['aspd']:.6f}")
        lines.append(f"  - zero-denominator warnings: {item['zero_denominator_check']['warnings']}")
    lines.append("")
    if report["errors"]:
        lines.append("## Errors")
        for err in report["errors"]:
            lines.append(f"- {err}")
    else:
        lines.append("## Result")
        lines.append("All validation gates passed.")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    report: Dict[str, Any] = {"files": [], "errors": []}
    for dataset in ["adult", "compas"]:
        for item in file_manifest(dataset):
            report["files"].append(item)
            if not item["exists"]:
                report["errors"].append(f"Missing required file: {item['file']}")

    if report["errors"]:
        write_reports(report)
        for err in report["errors"]:
            print(f"ERROR: {err}", file=sys.stderr)
        return 2

    adult = validate_adult()
    compas = validate_compas(seed=args.seed)
    metric_checks = [metric_self_check("adult"), metric_self_check("compas")]
    report.update({"adult": adult, "compas": compas, "metric_self_checks": metric_checks})
    report["errors"].extend([f"adult: {e}" for e in adult["errors"]])
    report["errors"].extend([f"compas: {e}" for e in compas["errors"]])

    write_reports(report)
    if report["errors"]:
        for err in report["errors"]:
            print(f"ERROR: {err}", file=sys.stderr)
        return 2

    print(f"Validation passed. Reports written under {RESULTS_DIR}")
    print(json.dumps({"adult": adult["train"], "compas": compas["cleaned_full"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
