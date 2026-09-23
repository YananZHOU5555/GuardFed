#!/usr/bin/env python3
"""Run GuardFed paper Table II/III reproduction experiments."""
from __future__ import annotations
import argparse, copy, csv, hashlib, json, math, random, subprocess, sys, time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import norm
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.data_loader import DatasetLoader  # noqa: E402

RESULTS_DIR = ROOT / "results" / "paper_tables"
RAW_PATH = RESULTS_DIR / "raw_results.jsonl"
TABLE_PATHS = {"adult": RESULTS_DIR / "table_adult.csv", "compas": RESULTS_DIR / "table_compas.csv"}
REPORT_PATH = RESULTS_DIR / "reproduction_report.md"
METHODS = ["FedAvg", "FairFed", "Median", "FLTrust", "FairGuard", "FLTrust+FairGuard", "GuardFed", "FLGMM", "FLAURA", "LayerGuard", "SmartFL", "FLTG", "FedDNA", "LASA", "Fed-NGA", "Huber-BRFL", "LoGoFair", "AdaAggRL", "FedAMM", "FedAA", "GuardFed-AD2", "GuardFed-AD2+"]
ATTACKS = ["Benign", "F Flip", "FOE", "S-DFA", "Sp-DFA"]
EXTRA_ATTACKS = ["FedSA"]
ALL_ATTACKS = ATTACKS + EXTRA_ATTACKS
DISTRIBUTIONS = {"IID": 5000.0, "non-IID": 5.0}
DATASETS = ["adult", "compas"]
METRICS = ["accuracy", "aeod", "aspd"]
SYNTH_ROOT_CACHE: Dict[Tuple[Any, ...], pd.DataFrame] = {}
FOE_SCALE = -0.5

@dataclass
class ExperimentConfig:
    seed: int = 123
    num_clients: int = 20
    num_malicious: int = 4
    local_epochs: int = 1
    batch_size: int = 256
    learning_rate: float = 0.005
    rounds: int = 70
    device: str = "cuda"
    server_ratio: float = 0.05
    synthetic_ratio: float = 0.0
    server_sampling: str = "stratified_sensitive"
    server_alpha: Optional[float] = None
    server_target_sensitive: Optional[int] = None
    server_target_label: Optional[int] = None
    synthetic_method: str = "gaussian_copula"
    synthetic_epochs: int = 50
    optimizer: str = "adam"
    include_sensitive_feature: bool = False
    aggregation_weighting: str = "count"
    fflip_mode: str = "invert"
    foe_mode: str = "state"
    sdfa_foe_mode: Optional[str] = None
    spdfa_foe_mode: Optional[str] = None
    fedsa_gain: float = 1.75
    fedsa_norm_ratio: float = 2.0
    fairguard_mode: str = "server_aeod"
    use_reweighting: bool = True
    fairfed_beta: float = 1.0
    trust_threshold: float = 0.2
    guardfed_fairness_lambda: float = 20.0
    act_fairness_budget: float = 0.06
    act_temperature: float = 0.35
    act_keep_ratio: float = 0.80
    act_fairness_metric: str = "aeod_aspd"
    act_anchor_drop: float = 0.005
    act_risk_weight: float = 1.0
    act_violation_weight: float = 1.0
    ad2_calibration_base_weight: float = 1.0
    ad2_calibration_budget: float = 0.06
    ad2_calibration_temperature: float = 0.03
    ad2_calibration_quantiles: int = 41
    ad2_score_clip: float = 5.0
    ad2_norm_clip_scale: float = 2.5
    ad2_calibration_max_acc_drop: float = 0.03
    ad2_calibration_objective: str = "acc_floor"
    ad2_calibration_enabled: bool = True
    ad2_norm_mode: str = "adaptive"
    ad2_utility_weight: float = 1.0
    ad2_centrality_weight: float = 0.35
    ad2_alignment_weight: float = 0.35
    ad2_plus_mode: str = "adaptive"
    experiment_suite: str = "main"
    experiment_tag: str = "default"
    ablation_component: str = "none"
    client_alpha: Optional[float] = None
    full_round_diagnostics: bool = False
    root_label_noise: float = 0.0
    root_sensitive_noise: float = 0.0
    compas_preprocessing_version: str = "legacy"
    celeba_cache_dir: str = ""
    celeba_train_limit: int = 0
    celeba_eval_limit: int = 0
    celeba_evaluation_split: str = "test"

    def __post_init__(self):
        if self.ablation_component not in {"none", "U", "C", "A", "F", "V", "N"}:
            raise ValueError(f"Unsupported ablation_component: {self.ablation_component}")
        if self.client_alpha is not None and (not math.isfinite(self.client_alpha) or self.client_alpha <= 0):
            raise ValueError("client_alpha must be finite and positive")
        for name in ("root_label_noise", "root_sensitive_noise"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be finite and in [0, 1]")
        if (self.root_label_noise or self.root_sensitive_noise) and self.synthetic_ratio:
            raise ValueError("Root noise currently requires synthetic_ratio=0")
        if self.compas_preprocessing_version not in {"legacy", "train_only"}:
            raise ValueError("Unsupported compas_preprocessing_version")
        if self.celeba_train_limit < 0 or self.celeba_eval_limit < 0:
            raise ValueError("CelebA subset limits must be nonnegative")
        if self.celeba_evaluation_split not in {"valid", "test"}:
            raise ValueError("CelebA evaluation split must be valid or test")

class SimpleMLP(nn.Module):
    def __init__(self, input_size: int, seed: int = 123):
        super().__init__()
        torch.manual_seed(seed)
        self.linear1 = nn.Linear(input_size, 16)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(16, 2)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.relu(self.linear1(x)))

def make_model(bundle: Dict[str, Any], config: ExperimentConfig, device: torch.device) -> nn.Module:
    if bundle.get("dataset") == "celeba":
        from src.celeba_data import CelebACNN
        return CelebACNN(config.seed).to(device)
    return SimpleMLP(bundle["num_features"], config.seed).to(device)


def set_seed(seed: int, deterministic_image: bool = False) -> None:
    if deterministic_image:
        import os
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic_image

def choose_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested device=cuda but CUDA is unavailable")
    if name == "auto": name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, sensitive: np.ndarray) -> Dict[str, Any]:
    y_true = np.asarray(y_true).astype(int); y_pred = np.asarray(y_pred).astype(int); sensitive = np.asarray(sensitive).astype(int)
    warnings: List[str] = []
    accuracy = float(np.mean(y_true == y_pred)) if len(y_true) else math.nan
    tprs, rates = {}, {}
    for g in [0, 1]:
        gm = sensitive == g; pm = gm & (y_true == 1)
        if int(pm.sum()) == 0:
            warnings.append(f"AEOD denominator is zero for sensitive group {g}"); tprs[g] = math.nan
        else: tprs[g] = float(np.mean(y_pred[pm] == 1))
        if int(gm.sum()) == 0:
            warnings.append(f"ASPD denominator is zero for sensitive group {g}"); rates[g] = math.nan
        else: rates[g] = float(np.mean(y_pred[gm] == 1))
    aeod = abs(tprs[0] - tprs[1]) if not (math.isnan(tprs[0]) or math.isnan(tprs[1])) else math.nan
    aspd = abs(rates[0] - rates[1]) if not (math.isnan(rates[0]) or math.isnan(rates[1])) else math.nan
    positive_rate = float(np.mean(y_pred == 1)) if len(y_pred) else math.nan
    majority_accuracy = float(max(np.mean(y_true == 0), np.mean(y_true == 1))) if len(y_true) else math.nan
    return {"accuracy": accuracy, "aeod": float(aeod) if not math.isnan(aeod) else math.nan,
            "aspd": float(aspd) if not math.isnan(aspd) else math.nan, "positive_rate": positive_rate,
            "majority_accuracy": majority_accuracy, "prediction_count": int(len(y_pred)), "warnings": warnings}

def evaluate_model(model: nn.Module, X: torch.Tensor, y: torch.Tensor, sensitive: np.ndarray, batch_size: int) -> Dict[str, Any]:
    model.eval(); preds=[]; truth=[]
    with torch.no_grad():
        for xb, yb in DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=False):
            preds.append(torch.argmax(model(xb), dim=1).cpu().numpy()); truth.append(yb.cpu().numpy())
    return compute_metrics(np.concatenate(truth), np.concatenate(preds), sensitive)

def model_margins(model: nn.Module, X: torch.Tensor, batch_size: int) -> np.ndarray:
    model.eval(); margins=[]
    dummy = torch.zeros(len(X), dtype=torch.long, device=X.device)
    with torch.no_grad():
        for xb, _yb in DataLoader(TensorDataset(X, dummy), batch_size=batch_size, shuffle=False):
            logits = model(xb)
            margins.append((logits[:, 1] - logits[:, 0]).detach().cpu().numpy())
    return np.concatenate(margins) if margins else np.array([], dtype=float)


def metrics_from_group_thresholds(y_true: np.ndarray, margins: np.ndarray, sensitive: np.ndarray, thresholds: Dict[int, float]) -> Dict[str, Any]:
    preds = np.zeros_like(y_true, dtype=int)
    for group in [0, 1]:
        mask = sensitive == group
        preds[mask] = (margins[mask] >= thresholds.get(group, 0.0)).astype(int)
    return compute_metrics(y_true, preds, sensitive)


def threshold_candidates(margins: np.ndarray, num_quantiles: int) -> List[float]:
    if len(margins) == 0:
        return [0.0]
    qn = max(7, int(num_quantiles))
    qs = np.quantile(margins, np.linspace(0.02, 0.98, qn)).astype(float).tolist()
    vals = qs + [0.0, float(np.min(margins) - 1e-6), float(np.max(margins) + 1e-6)]
    return sorted(set(round(v, 8) for v in vals))


def calibration_risk(metrics: Dict[str, Any]) -> float:
    aeod = float(metrics["aeod"]) if not math.isnan(float(metrics["aeod"])) else 1.0
    aspd = float(metrics["aspd"]) if not math.isnan(float(metrics["aspd"])) else 1.0
    return 0.5 * (aeod + aspd)


def fit_group_thresholds(model: nn.Module, bundle: Dict[str, Any], config: ExperimentConfig) -> Tuple[Dict[int, float], Dict[str, Any]]:
    margins = model_margins(model, bundle["server_X"], config.batch_size)
    y = bundle["server_y"].detach().cpu().numpy()
    s = np.asarray(bundle["server_sensitive"], dtype=int)
    candidates = {group: threshold_candidates(margins[s == group], config.ad2_calibration_quantiles) for group in [0, 1]}
    base_metrics = metrics_from_group_thresholds(y, margins, s, {0: 0.0, 1: 0.0})
    base_risk = calibration_risk(base_metrics)
    temp = max(config.ad2_calibration_temperature, 1e-6)
    adaptive_lambda = config.ad2_calibration_base_weight * math.log1p(math.exp((base_risk - config.ad2_calibration_budget) / temp))
    best_thresholds = {0: 0.0, 1: 0.0}
    best_obj = -1e18
    best_metrics: Dict[str, Any] = {}
    best_risk = 1.0
    base_acc = float(base_metrics["accuracy"]) if math.isfinite(float(base_metrics["accuracy"])) else 0.0
    acc_floor = max(0.0, base_acc - max(0.0, config.ad2_calibration_max_acc_drop))
    for t0 in candidates[0]:
        for t1 in candidates[1]:
            thresholds = {0: float(t0), 1: float(t1)}
            metrics = metrics_from_group_thresholds(y, margins, s, thresholds)
            fair_avg = calibration_risk(metrics)
            acc = float(metrics["accuracy"]) if math.isfinite(float(metrics["accuracy"])) else 0.0
            violation = max(0.0, fair_avg - config.ad2_calibration_budget)
            if config.ad2_calibration_objective == "original":
                obj = acc - fair_avg - adaptive_lambda * violation
            else:
                acc_shortfall = max(0.0, acc_floor - acc)
                if acc >= acc_floor:
                    obj = 10.0 - fair_avg - adaptive_lambda * violation + 0.01 * acc
                else:
                    obj = acc - fair_avg - adaptive_lambda * violation - 10.0 * acc_shortfall
            if obj > best_obj or (math.isclose(obj, best_obj) and acc > float(best_metrics.get("accuracy", -1.0))):
                best_obj = obj
                best_thresholds = thresholds
                best_metrics = metrics
                best_risk = fair_avg
    return best_thresholds, {
        "server_calibration_metrics": {k: best_metrics.get(k) for k in METRICS},
        "server_calibration_score": best_obj,
        "server_calibration_risk": best_risk,
        "server_base_calibration_metrics": {k: base_metrics.get(k) for k in METRICS},
        "server_base_calibration_risk": base_risk,
        "server_adaptive_lambda": adaptive_lambda,
        "server_calibration_budget": config.ad2_calibration_budget,
        "server_calibration_acc_floor": acc_floor,
        "server_calibration_max_acc_drop": config.ad2_calibration_max_acc_drop,
        "server_calibration_objective": config.ad2_calibration_objective,
    }


def evaluate_model_calibrated(model: nn.Module, bundle: Dict[str, Any], config: ExperimentConfig) -> Dict[str, Any]:
    thresholds, info = fit_group_thresholds(model, bundle, config)
    margins = model_margins(model, bundle["X_test"], config.batch_size)
    y = bundle["y_test"].detach().cpu().numpy()
    s = np.asarray(bundle["test_sensitive"], dtype=int)
    metrics = metrics_from_group_thresholds(y, margins, s, thresholds)
    metrics["calibration_thresholds"] = thresholds
    metrics["calibration_info"] = info
    return metrics


def evaluate_for_reporting(method: str, model: nn.Module, bundle: Dict[str, Any], config: ExperimentConfig) -> Dict[str, Any]:
    if method in {"GuardFed-AD2", "GuardFed-AD2+", "LoGoFair"} and config.ad2_calibration_enabled:
        return evaluate_model_calibrated(model, bundle, config)
    return evaluate_model(model, bundle["X_test"], bundle["y_test"], bundle["test_sensitive"], config.batch_size)

def clone_state(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in model.state_dict().items()}

def state_delta(local: Dict[str, torch.Tensor], global_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: local[k].detach().clone() - global_state[k].detach().clone() for k in global_state}

def zero_like(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: torch.zeros_like(v) for k, v in state.items()}

def vectorize(update: Dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([v.reshape(-1) for v in update.values()])

def devectorize(vec: torch.Tensor, template: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    pos = 0
    for k, v in template.items():
        n = v.numel()
        out[k] = vec[pos:pos+n].reshape_as(v).to(device=v.device, dtype=v.dtype)
        pos += n
    return out

def update_norm(update: Dict[str, torch.Tensor]) -> float:
    return float(torch.linalg.vector_norm(vectorize(update)).cpu())

def cosine_between(a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor]) -> float:
    av, bv = vectorize(a), vectorize(b)
    if float(torch.linalg.vector_norm(av)) == 0.0 or float(torch.linalg.vector_norm(bv)) == 0.0: return math.nan
    return float(F.cosine_similarity(av.unsqueeze(0), bv.unsqueeze(0)).item())

def apply_update(model: nn.Module, update: Dict[str, torch.Tensor]) -> None:
    state = clone_state(model)
    for k in state: state[k] = state[k] + update[k]
    model.load_state_dict(state)

def weighted_ce(logits: torch.Tensor, y: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    losses = F.cross_entropy(logits, y, reduction="none")
    return torch.sum(losses * weights) / torch.clamp(weights.sum(), min=1e-12)


def make_optimizer(model: nn.Module, config: ExperimentConfig):
    if config.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    if config.optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=config.learning_rate)
    raise ValueError(f"Unsupported optimizer: {config.optimizer}")


def gaussian_copula_augment(
    root_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    feature_cols: List[str],
    label_col: str,
    sensitive_col: str,
    n_synth: int,
    seed: int,
) -> pd.DataFrame:
    """Generate tabular synthetic root rows with an empirical Gaussian Copula.

    The model is fitted on the clean 1% root rows. Values are projected back to
    the empirical support of the full encoded training data to keep categorical,
    sensitive, and label columns valid.
    """
    cols = feature_cols + [label_col]
    if sensitive_col not in cols:
        cols = feature_cols + [sensitive_col, label_col]
    fit = root_df[cols].astype(float).reset_index(drop=True)
    support = reference_df[cols].reset_index(drop=True)
    if n_synth <= 0 or len(fit) < 4:
        return fit.iloc[0:0].copy()
    rng = np.random.default_rng(seed)
    z_cols = []
    for col in cols:
        ranks = fit[col].rank(method="average").to_numpy()
        u = np.clip((ranks - 0.5) / len(fit), 1e-6, 1 - 1e-6)
        z_cols.append(norm.ppf(u))
    z = np.vstack(z_cols).T
    corr = np.corrcoef(z, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    corr = (corr + corr.T) / 2.0
    corr += np.eye(len(cols)) * 1e-6
    try:
        sampled_z = rng.multivariate_normal(np.zeros(len(cols)), corr, size=n_synth)
    except np.linalg.LinAlgError:
        sampled_z = rng.multivariate_normal(np.zeros(len(cols)), np.eye(len(cols)), size=n_synth)
    sampled_u = np.clip(norm.cdf(sampled_z), 0.0, 1.0)
    out = pd.DataFrame(index=range(n_synth), columns=cols)
    for j, col in enumerate(cols):
        values = np.sort(support[col].to_numpy())
        quantile_idx = np.clip((sampled_u[:, j] * (len(values) - 1)).round().astype(int), 0, len(values) - 1)
        out[col] = values[quantile_idx]
        if col == label_col or col == sensitive_col or not pd.api.types.is_float_dtype(support[col]):
            out[col] = out[col].round().astype(int)
    return out


def _integer_like(series: pd.Series) -> bool:
    vals = series.dropna().to_numpy()
    if len(vals) == 0:
        return True
    return bool(np.allclose(vals, np.round(vals)))


def _project_synthetic_columns(out: pd.DataFrame, support: pd.DataFrame, label_col: str, sensitive_col: str) -> pd.DataFrame:
    projected = out.copy()
    for col in projected.columns:
        if col not in support.columns:
            continue
        if col == label_col or col == sensitive_col or _integer_like(support[col]) or support[col].nunique(dropna=True) <= 20:
            values = np.sort(support[col].dropna().unique())
            if len(values) == 0:
                continue
            raw = projected[col].astype(float).to_numpy()
            nearest_idx = np.abs(raw[:, None] - values[None, :]).argmin(axis=1)
            projected[col] = values[nearest_idx].astype(int if _integer_like(support[col]) else float)
        else:
            lo = float(support[col].min())
            hi = float(support[col].max())
            projected[col] = projected[col].astype(float).clip(lo, hi)
    return projected


def bootstrap_augment(root_df: pd.DataFrame, cols: List[str], n_synth: int, seed: int) -> pd.DataFrame:
    if n_synth <= 0 or len(root_df) == 0:
        return root_df.iloc[0:0][cols].copy()
    return root_df[cols].sample(n=n_synth, replace=True, random_state=seed).reset_index(drop=True)


def noisy_bootstrap_augment(root_df: pd.DataFrame, reference_df: pd.DataFrame, cols: List[str], label_col: str, sensitive_col: str, n_synth: int, seed: int) -> pd.DataFrame:
    if n_synth <= 0 or len(root_df) == 0:
        return root_df.iloc[0:0][cols].copy()
    rng = np.random.default_rng(seed)
    out = bootstrap_augment(root_df, cols, n_synth, seed)
    for col in cols:
        if col in {label_col, sensitive_col} or _integer_like(reference_df[col]) or reference_df[col].nunique(dropna=True) <= 20:
            continue
        sigma = float(root_df[col].std(ddof=0))
        if math.isfinite(sigma) and sigma > 0:
            out[col] = out[col].astype(float) + rng.normal(0.0, 0.05 * sigma, size=n_synth)
    return _project_synthetic_columns(out, reference_df[cols], label_col, sensitive_col)


def smote_augment(root_df: pd.DataFrame, reference_df: pd.DataFrame, cols: List[str], label_col: str, sensitive_col: str, n_synth: int, seed: int) -> pd.DataFrame:
    if n_synth <= 0 or len(root_df) == 0:
        return root_df.iloc[0:0][cols].copy()
    rng = np.random.default_rng(seed)
    strata = [g for _, g in root_df.groupby([sensitive_col, label_col], dropna=False) if len(g) > 0]
    if not strata:
        return bootstrap_augment(root_df, cols, n_synth, seed)
    rows = []
    for _ in range(n_synth):
        group = strata[int(rng.integers(0, len(strata)))]
        if len(group) == 1:
            row = group.iloc[0][cols].astype(float).copy()
        else:
            pair = group.sample(n=2, replace=True, random_state=int(rng.integers(0, 2**31 - 1)))[cols].astype(float)
            lam = float(rng.random())
            row = pair.iloc[0] * lam + pair.iloc[1] * (1.0 - lam)
        rows.append(row)
    out = pd.DataFrame(rows, columns=cols)
    return _project_synthetic_columns(out, reference_df[cols], label_col, sensitive_col)


def pca_gaussian_augment(root_df: pd.DataFrame, reference_df: pd.DataFrame, cols: List[str], label_col: str, sensitive_col: str, n_synth: int, seed: int) -> pd.DataFrame:
    if n_synth <= 0 or len(root_df) < 3:
        return root_df.iloc[0:0][cols].copy()
    rng = np.random.default_rng(seed)
    fit = root_df[cols].astype(float).to_numpy()
    mean = fit.mean(axis=0)
    cov = np.cov(fit, rowvar=False)
    cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)
    shrink = np.eye(len(cols)) * max(1e-6, float(np.trace(cov)) / max(1, len(cols)) * 1e-3)
    try:
        sampled = rng.multivariate_normal(mean, cov + shrink, size=n_synth)
    except np.linalg.LinAlgError:
        sampled = rng.multivariate_normal(mean, np.diag(np.diag(cov) + 1e-6), size=n_synth)
    out = pd.DataFrame(sampled, columns=cols)
    return _project_synthetic_columns(out, reference_df[cols], label_col, sensitive_col)


def ctgan_augment(root_df: pd.DataFrame, reference_df: pd.DataFrame, cols: List[str], label_col: str, sensitive_col: str, n_synth: int, seed: int, epochs: int, model_name: str) -> pd.DataFrame:
    if n_synth <= 0 or len(root_df) < 10:
        return root_df.iloc[0:0][cols].copy()
    try:
        from ctgan import CTGAN, TVAE
    except Exception as exc:
        raise RuntimeError("synthetic_method=ctgan/tvae requires the ctgan package in the active venv") from exc
    import torch as _torch
    _torch.manual_seed(seed)
    np.random.seed(seed)
    fit = root_df[cols].reset_index(drop=True).copy()
    discrete_columns = [col for col in cols if col in {label_col, sensitive_col} or _integer_like(reference_df[col]) or reference_df[col].nunique(dropna=True) <= 20]
    batch_size = max(10, min(500, int(math.ceil(len(fit) / 10.0) * 10)))
    if model_name == "ctgan":
        model = CTGAN(epochs=int(epochs), batch_size=batch_size, pac=1, verbose=False, enable_gpu=True)
    elif model_name == "tvae":
        model = TVAE(epochs=int(epochs), batch_size=batch_size, verbose=False, enable_gpu=True)
    else:
        raise ValueError(f"Unsupported CTGAN-family model: {model_name}")
    model.fit(fit, discrete_columns=discrete_columns)
    out = model.sample(n_synth)
    for col in cols:
        if col not in out.columns:
            out[col] = fit[col].sample(n=n_synth, replace=True, random_state=seed).reset_index(drop=True)
    out = out[cols].reset_index(drop=True)
    return _project_synthetic_columns(out, reference_df[cols], label_col, sensitive_col)


def make_synthetic_root(
    method: str,
    root_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    feature_cols: List[str],
    label_col: str,
    sensitive_col: str,
    n_synth: int,
    seed: int,
    epochs: int = 50,
) -> pd.DataFrame:
    cols = list(dict.fromkeys(feature_cols + [sensitive_col, label_col]))
    if n_synth <= 0 or method == "none":
        return root_df.iloc[0:0][cols].copy()
    if method == "gaussian_copula":
        return gaussian_copula_augment(root_df, reference_df, feature_cols, label_col, sensitive_col, n_synth, seed)[cols]
    if method == "bootstrap":
        return bootstrap_augment(root_df, cols, n_synth, seed)
    if method == "noisy_bootstrap":
        return noisy_bootstrap_augment(root_df, reference_df, cols, label_col, sensitive_col, n_synth, seed)
    if method == "smote":
        return smote_augment(root_df, reference_df, cols, label_col, sensitive_col, n_synth, seed)
    if method == "pca_gaussian":
        return pca_gaussian_augment(root_df, reference_df, cols, label_col, sensitive_col, n_synth, seed)
    if method in {"ctgan", "tvae"}:
        return ctgan_augment(root_df, reference_df, cols, label_col, sensitive_col, n_synth, seed, epochs, method)
    raise ValueError(f"Unsupported synthetic_method: {method}")


def compute_reweighing_weights(df: pd.DataFrame, sensitive_col: str, label_col: str) -> Dict[Tuple[int, int], float]:
    total = len(df); p_s = df[sensitive_col].value_counts(normalize=True).to_dict(); p_y = df[label_col].value_counts(normalize=True).to_dict(); p_obs = (df.groupby([sensitive_col, label_col]).size() / total).to_dict(); out={}
    for s in [0, 1]:
        for y in [0, 1]:
            exp = float(p_s.get(s, 0.0) * p_y.get(y, 0.0)); obs = float(p_obs.get((s, y), 0.0)); out[(s, y)] = exp / obs if obs > 0 else 1.0
    return out

def attack_types_for_client(attack: str, cid: int, malicious_ids: Sequence[int]) -> List[str]:
    if attack == "Benign" or cid not in malicious_ids: return []
    if attack == "F Flip": return ["fflip"]
    if attack in {"FOE", "FedSA"}: return ["foe"]
    if attack == "S-DFA": return ["fflip", "foe"]
    if attack == "Sp-DFA": return ["fflip"] if list(malicious_ids).index(cid) < len(malicious_ids)//2 else ["foe"]
    raise ValueError(f"Unknown attack {attack}")

def client_runtime_data(
    raw: Dict[str, Any],
    cid: int,
    attack: str,
    malicious_ids: Sequence[int],
    rw: Dict[Tuple[int,int], float],
    device: torch.device,
    fflip_mode: str,
    foe_mode: str,
    sdfa_foe_mode: Optional[str],
    spdfa_foe_mode: Optional[str],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    image_input = raw["X"].ndim == 4
    data_device = torch.device("cpu") if image_input else device
    X = raw["X"] if image_input else raw["X"].detach().clone().to(device)
    y = raw["y"].detach().clone().to(data_device)
    sensitive = np.asarray(raw["sensitive"], dtype=int).copy()
    orig_s = sensitive.copy()
    orig_y = y.cpu().numpy().astype(int).copy()
    ats = attack_types_for_client(attack, cid, malicious_ids)
    active_foe_mode = foe_mode
    if attack == "FedSA":
        active_foe_mode = "fedsa"
    if attack == "S-DFA" and sdfa_foe_mode is not None:
        active_foe_mode = sdfa_foe_mode
    if attack == "Sp-DFA" and spdfa_foe_mode is not None:
        active_foe_mode = spdfa_foe_mode
    audit: Dict[str, Any] = {"client_id": cid, "is_malicious": cid in malicious_ids, "attack_types": ats, "samples": int(len(y)), "foe_mode": active_foe_mode if "foe" in ats else None}
    if "fflip" in ats:
        y_np = y.cpu().numpy().astype(int)
        if len(sensitive) > 0:
            if fflip_mode == "invert":
                sensitive = 1 - sensitive
                requires_full_flip = True
            elif fflip_mode == "label_conditioned":
                sensitive = np.where(y_np == 0, 1, 0).astype(int)
                requires_full_flip = False
            elif fflip_mode == "label_conditioned_reverse":
                sensitive = np.where(y_np == 0, 0, 1).astype(int)
                requires_full_flip = False
            elif fflip_mode == "all_privileged":
                sensitive = np.ones_like(sensitive)
                requires_full_flip = False
            elif fflip_mode == "all_unprivileged":
                sensitive = np.zeros_like(sensitive)
                requires_full_flip = False
            else:
                raise ValueError(f"Unsupported fflip_mode: {fflip_mode}")
            changed = sensitive != orig_s
            audit["fflip_changed"] = int(np.sum(changed))
            audit["fflip_ratio"] = float(np.mean(changed))
            audit["fflip_mode"] = fflip_mode
            audit["fflip_requires_full_flip"] = requires_full_flip
            audit["fflip_overwrite_ratio"] = 1.0
            if len(sensitive) > 1 and np.std(orig_s) > 0 and np.std(y_np) > 0:
                audit["fflip_label_corr_before"] = float(np.corrcoef(orig_s, y_np)[0, 1])
            else:
                audit["fflip_label_corr_before"] = math.nan
            if len(sensitive) > 1 and np.std(sensitive) > 0 and np.std(y_np) > 0:
                audit["fflip_label_corr_after"] = float(np.corrcoef(sensitive, y_np)[0, 1])
            else:
                audit["fflip_label_corr_after"] = math.nan
            sens_idx = raw.get("sensitive_feature_index")
            if sens_idx is not None:
                X[:, int(sens_idx)] = torch.tensor(sensitive, dtype=X.dtype, device=device)
                audit["fflip_feature_column_changed"] = int(np.sum(changed))
        else:
            audit["fflip_changed"] = 0
            audit["fflip_ratio"] = math.nan
            audit["fflip_mode"] = fflip_mode
            audit["fflip_requires_full_flip"] = fflip_mode == "invert"
        audit["label_changed_count"] = int(np.sum(y.cpu().numpy().astype(int) != orig_y))
    y_np = y.cpu().numpy().astype(int)
    sw = np.asarray([rw.get((int(s), int(label)), 1.0) for s, label in zip(sensitive, y_np)], dtype="float32")
    return {"cid": cid, "X": X, "y": y, "sensitive": sensitive, "weights": torch.tensor(sw, dtype=torch.float32, device=data_device), "n": int(len(y)), "attack_types": ats, "attack": attack, "foe_mode": active_foe_mode}, audit

def train_local_model(global_model: nn.Module, client: Dict[str, Any], config: ExperimentConfig) -> Dict[str, torch.Tensor]:
    if client["n"] == 0: return clone_state(global_model)
    local = copy.deepcopy(global_model); local.train(); opt = make_optimizer(local, config)
    loader = DataLoader(TensorDataset(client["X"], client["y"], client["weights"]), batch_size=min(config.batch_size, max(1, client["n"])), shuffle=True)
    for _ in range(config.local_epochs):
        for xb, yb, wb in loader:
            if xb.ndim == 4:
                yb, wb = yb.to(next(local.parameters()).device), wb.to(next(local.parameters()).device)
            opt.zero_grad(set_to_none=True); loss = weighted_ce(local(xb), yb, wb); loss.backward(); opt.step()
    return clone_state(local)

def train_server_update(global_model: nn.Module, bundle: Dict[str, Any], config: ExperimentConfig) -> Dict[str, torch.Tensor]:
    X, y = bundle["server_X"], bundle["server_y"]
    if len(y) == 0: return zero_like(clone_state(global_model))
    local = copy.deepcopy(global_model); local.train(); opt = make_optimizer(local, config)
    for xb, yb in DataLoader(TensorDataset(X, y), batch_size=min(config.batch_size, len(y)), shuffle=True):
        if xb.ndim == 4:
            yb = yb.to(next(local.parameters()).device)
        opt.zero_grad(set_to_none=True); loss = F.cross_entropy(local(xb), yb); loss.backward(); opt.step()
    return state_delta(clone_state(local), clone_state(global_model))

def evaluate_state_on_server(state: Dict[str, torch.Tensor], input_size: int, bundle: Dict[str, Any], config: ExperimentConfig, device: torch.device) -> Dict[str, Any]:
    model = make_model(bundle, config, device); model.load_state_dict(state)
    return evaluate_model(model, bundle["server_X"], bundle["server_y"], bundle["server_sensitive"], config.batch_size)

def fairguard_select(fairness: List[float], seed: int) -> List[int]:
    vals = np.asarray([f if not math.isnan(float(f)) else 1.0 for f in fairness], dtype=float)
    if len(vals) <= 2 or len(np.unique(np.round(vals, 8))) < 2: return list(range(len(fairness)))
    labels = KMeans(n_clusters=2, random_state=seed, n_init=10).fit_predict(vals.reshape(-1, 1)); centers = [float(vals[labels == i].mean()) for i in [0, 1]]; keep = int(np.argmin(centers))
    selected = [i for i, label in enumerate(labels) if int(label) == keep]
    return selected or list(range(len(fairness)))

def weighted_average(updates: List[Dict[str, torch.Tensor]], weights: List[float]) -> Dict[str, torch.Tensor]:
    total = float(sum(weights));
    if total <= 0: weights = [1.0] * len(updates); total = float(len(updates))
    out = {k: torch.zeros_like(v) for k, v in updates[0].items()}
    for u, w in zip(updates, weights):
        for k in out: out[k] = out[k] + u[k] * (float(w) / total)
    return out

def median_update(updates: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return {k: torch.median(torch.stack([u[k] for u in updates], dim=0), dim=0).values for k in updates[0]}


def update_distance(a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor]) -> float:
    return float(torch.linalg.vector_norm(vectorize({k: a[k] - b[k] for k in a})).cpu())


def robust_median_mad(values: Sequence[float]) -> Tuple[float, float]:
    arr = np.asarray([float(v) for v in values], dtype=float)
    if arr.size == 0:
        return 0.0, 1.0
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    return med, max(1.4826 * mad, 1e-6)


def sigmoid(x: float) -> float:
    if x >= 40:
        return 1.0
    if x <= -40:
        return 0.0
    return float(1.0 / (1.0 + math.exp(-x)))


def select_top_scores(scores: Sequence[float], keep_ratio: float, min_keep: int = 1) -> List[int]:
    n = len(scores)
    keep = max(min_keep, min(n, int(math.ceil(n * keep_ratio))))
    order = np.argsort(np.asarray(scores, dtype=float))
    return sorted([int(i) for i in order[-keep:]])


def softmax_weights(scores: Sequence[float], temperature: float) -> List[float]:
    temp = max(float(temperature), 1e-6)
    arr = np.asarray(scores, dtype=float) / temp
    arr = arr - np.max(arr)
    w = np.exp(arr)
    if not np.isfinite(w).all() or float(w.sum()) <= 0:
        return [1.0] * len(scores)
    return [float(x) for x in w]


def flgmm_select_from_distances(distances: Sequence[float]) -> Tuple[List[int], Dict[str, Any]]:
    """Small 1D two-component GMM EM used for FLGMM-style adaptive filtering."""
    x = np.asarray(distances, dtype=float)
    n = len(x)
    if n <= 2 or float(np.max(x) - np.min(x)) < 1e-12:
        return list(range(n)), {"distances": x.tolist(), "gmm_means": [], "gmm_selected_component": "all"}
    means = np.array([float(np.min(x)), float(np.max(x))], dtype=float)
    vars_ = np.array([float(np.var(x) + 1e-6), float(np.var(x) + 1e-6)], dtype=float)
    pis = np.array([0.5, 0.5], dtype=float)
    for _ in range(30):
        probs = []
        for k in range(2):
            coef = 1.0 / math.sqrt(2.0 * math.pi * vars_[k])
            probs.append(pis[k] * coef * np.exp(-0.5 * ((x - means[k]) ** 2) / vars_[k]))
        probs = np.vstack(probs).T + 1e-12
        resp = probs / probs.sum(axis=1, keepdims=True)
        nk = resp.sum(axis=0) + 1e-12
        pis = nk / n
        means = (resp * x[:, None]).sum(axis=0) / nk
        vars_ = (resp * ((x[:, None] - means) ** 2)).sum(axis=0) / nk + 1e-6
    benign_comp = int(np.argmin(means))
    selected = [int(i) for i in range(n) if int(np.argmax(resp[i])) == benign_comp]
    if len(selected) < max(1, n // 2):
        selected = [int(i) for i in np.argsort(x)[:max(1, n - max(1, n // 5))]]
    return sorted(selected), {"distances": x.tolist(), "gmm_means": [float(v) for v in means], "gmm_selected_component": benign_comp}


def flgmm_aggregate(updates: List[Dict[str, torch.Tensor]], counts: List[int]) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    center = median_update(updates)
    distances = [update_distance(u, center) for u in updates]
    selected, info = flgmm_select_from_distances(distances)
    med, scale = robust_median_mad([distances[i] for i in selected])
    weights = [1.0 / (1e-6 + max(0.0, distances[i] - med + scale)) for i in selected]
    return weighted_average([updates[i] for i in selected], weights), {"selected_clients": selected, "method_core": "FLGMM-lite distance GMM", **info}


def flaura_aggregate(updates: List[Dict[str, torch.Tensor]], server_update: Dict[str, torch.Tensor], config: ExperimentConfig) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    # Core reproduction of FLAURA's adaptive trust idea: robust center + MMD-like distance boundary + hard/soft aggregation.
    center = median_update(updates)
    distances = [update_distance(u, center) for u in updates]
    med, scale = robust_median_mad(distances)
    boundary = med + scale
    selected = [i for i, d in enumerate(distances) if d <= boundary]
    min_keep = max(1, len(updates) - config.num_malicious)
    if len(selected) < min_keep:
        selected = [int(i) for i in np.argsort(distances)[:min_keep]]
    raw_scores = [-(distances[i] - med) / scale for i in selected]
    weights = softmax_weights(raw_scores, config.act_temperature)
    return weighted_average([updates[i] for i in selected], weights), {"selected_clients": selected, "method_core": "FLAURA-lite adaptive hard-soft trust", "distance_median": med, "distance_scale": scale, "boundary": boundary}


def layerguard_aggregate(updates: List[Dict[str, torch.Tensor]], config: ExperimentConfig) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    center = median_update(updates)
    scores: List[float] = []
    layer_scores: List[List[float]] = []
    for update in updates:
        local_scores = []
        for k in update:
            a = update[k].reshape(-1); b = center[k].reshape(-1)
            an = float(torch.linalg.vector_norm(a)); bn = float(torch.linalg.vector_norm(b))
            if an == 0.0 or bn == 0.0:
                local_scores.append(0.0)
            else:
                local_scores.append(max(0.0, float(torch.dot(a, b) / (an * bn))))
        layer_scores.append(local_scores)
        scores.append(float(np.mean(local_scores)) if local_scores else 0.0)
    selected = select_top_scores(scores, config.act_keep_ratio, min_keep=max(1, len(updates) - config.num_malicious))
    weights = softmax_weights([scores[i] for i in selected], config.act_temperature)
    return weighted_average([updates[i] for i in selected], weights), {"selected_clients": selected, "method_core": "LayerGuard-lite layer-wise similarity", "credibility_scores": scores, "layer_scores": layer_scores}


def smartfl_aggregate(updates: List[Dict[str, torch.Tensor]], config: ExperimentConfig) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    # SmartFL-core: majority pseudo-direction followed by positive-correlation retrieval.
    n = len(updates)
    vecs = [vectorize(u) for u in updates]
    unit = []
    for v in vecs:
        norm_v = torch.linalg.vector_norm(v)
        unit.append(v / torch.clamp(norm_v, min=1e-12))
    sims = torch.stack(unit) @ torch.stack(unit).T
    mean_sims = sims.mean(dim=1).detach().cpu().numpy().astype(float).tolist()
    min_keep = max(1, n - config.num_malicious)
    coarse = select_top_scores(mean_sims, keep_ratio=min_keep / max(n, 1), min_keep=min_keep)
    pseudo = weighted_average([updates[i] for i in coarse], [1.0] * len(coarse))
    retrieval_scores = []
    for update in updates:
        cos = cosine_between(pseudo, update)
        retrieval_scores.append(max(0.0, 0.0 if math.isnan(cos) else cos))
    selected = [i for i, score in enumerate(retrieval_scores) if score > 0.0]
    if len(selected) < min_keep:
        selected = select_top_scores(retrieval_scores, keep_ratio=min_keep / max(n, 1), min_keep=min_keep)
    weights = [max(1e-6, retrieval_scores[i]) for i in selected]
    return weighted_average([updates[i] for i in selected], weights), {"selected_clients": selected, "method_core": "SmartFL-core majority pseudo-direction", "coarse_clients": coarse, "retrieval_scores": retrieval_scores}


def fltg_aggregate(updates: List[Dict[str, torch.Tensor]], server_update: Dict[str, torch.Tensor], config: ExperimentConfig) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    # FLTG-core: ReLU-clipped server-angle filtering, reference-angle weighting, and norm normalization.
    n = len(updates)
    server_scores = []
    for update in updates:
        cos = cosine_between(server_update, update)
        server_scores.append(max(0.0, 0.0 if math.isnan(cos) else cos))
    min_keep = max(1, n - config.num_malicious)
    selected = [i for i, score in enumerate(server_scores) if score > 0.0]
    if len(selected) < min_keep:
        selected = select_top_scores(server_scores, keep_ratio=min_keep / max(n, 1), min_keep=min_keep)
    ref_idx = max(selected, key=lambda i: server_scores[i]) if selected else 0
    root_norm = update_norm(server_update)
    scaled = []
    weights = []
    for i in selected:
        ref_cos = cosine_between(updates[ref_idx], updates[i])
        ref_score = max(0.0, 0.0 if math.isnan(ref_cos) else ref_cos)
        norm_i = update_norm(updates[i])
        scale = root_norm / (norm_i + 1e-12) if root_norm > 0 and norm_i > 0 else 1.0
        scaled.append({k: v * scale for k, v in updates[i].items()})
        weights.append(max(1e-6, server_scores[i] * (0.5 + 0.5 * ref_score)))
    return weighted_average(scaled, weights), {"selected_clients": selected, "method_core": "FLTG-core angle defense", "server_angle_scores": server_scores, "reference_client": ref_idx}


def feddna_aggregate(updates: List[Dict[str, torch.Tensor]], server_update: Dict[str, torch.Tensor], config: ExperimentConfig) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    # FedDNA-core: update fingerprinting with MAD-adaptive anomaly thresholding.
    center = median_update(updates)
    distances = [update_distance(u, center) for u in updates]
    norms = [update_norm(u) for u in updates]
    server_cos = []
    for update in updates:
        cos = cosine_between(server_update, update)
        server_cos.append(0.0 if math.isnan(cos) else cos)
    d_med, d_scale = robust_median_mad(distances)
    n_med, n_scale = robust_median_mad(norms)
    c_med, c_scale = robust_median_mad(server_cos)
    anomaly = []
    for d, nrm, cos in zip(distances, norms, server_cos):
        z_d = abs(d - d_med) / d_scale
        z_n = abs(nrm - n_med) / n_scale
        z_c = max(0.0, (c_med - cos) / c_scale)
        anomaly.append(float(z_d + z_n + z_c))
    a_med, a_scale = robust_median_mad(anomaly)
    threshold = a_med + a_scale
    selected = [i for i, a in enumerate(anomaly) if a <= threshold]
    min_keep = max(1, len(updates) - config.num_malicious)
    if len(selected) < min_keep:
        selected = [int(i) for i in np.argsort(np.asarray(anomaly))[:min_keep]]
    weights = [math.exp(-anomaly[i]) for i in selected]
    return weighted_average([updates[i] for i in selected], weights), {"selected_clients": selected, "method_core": "FedDNA-core fingerprint MAD", "fingerprint_anomaly": anomaly, "mad_threshold": threshold}


def sparsify_tensor(t: torch.Tensor, keep_ratio: float) -> torch.Tensor:
    flat = t.reshape(-1)
    if flat.numel() <= 1:
        return t.clone()
    k = max(1, min(flat.numel(), int(math.ceil(flat.numel() * keep_ratio))))
    threshold = torch.topk(torch.abs(flat), k).values[-1]
    return t * (torch.abs(t) >= threshold).to(dtype=t.dtype)


def sparsify_update(update: Dict[str, torch.Tensor], keep_ratio: float) -> Dict[str, torch.Tensor]:
    return {k: sparsify_tensor(v, keep_ratio) for k, v in update.items()}


def lasa_aggregate(updates: List[Dict[str, torch.Tensor]], config: ExperimentConfig) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    # LASA-core: per-update sparsification followed by layer-wise magnitude/direction filtering.
    sparse_ratio = min(max(config.act_keep_ratio, 0.10), 1.0)
    sparse_updates = [sparsify_update(u, sparse_ratio) for u in updates]
    out: Dict[str, torch.Tensor] = {}
    selected_by_layer: Dict[str, List[int]] = {}
    scores_by_layer: Dict[str, List[float]] = {}
    for k in sparse_updates[0]:
        layer_vals = [u[k] for u in sparse_updates]
        stacked = torch.stack(layer_vals, dim=0)
        center = torch.median(stacked, dim=0).values
        norms = [float(torch.linalg.vector_norm(v.reshape(-1)).cpu()) for v in layer_vals]
        med_norm, scale_norm = robust_median_mad(norms)
        scores = []
        for v, norm_v in zip(layer_vals, norms):
            a = v.reshape(-1); b = center.reshape(-1)
            an = float(torch.linalg.vector_norm(a)); bn = float(torch.linalg.vector_norm(b))
            cos = 0.0 if an == 0.0 or bn == 0.0 else float(torch.dot(a, b) / (an * bn))
            mag_score = sigmoid(-(abs(norm_v - med_norm) / scale_norm))
            scores.append(max(0.0, cos) * mag_score)
        selected = select_top_scores(scores, config.act_keep_ratio, min_keep=max(1, len(updates) - config.num_malicious))
        out[k] = torch.mean(torch.stack([layer_vals[i] for i in selected], dim=0), dim=0)
        selected_by_layer[k] = selected
        scores_by_layer[k] = scores
    return out, {"selected_clients": sorted(set(i for xs in selected_by_layer.values() for i in xs)), "method_core": "LASA-core layer-adaptive sparsified aggregation", "selected_by_layer": selected_by_layer, "layer_scores": scores_by_layer, "sparse_ratio": sparse_ratio}



def fed_nga_aggregate(updates: List[Dict[str, torch.Tensor]], counts: List[int]) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """Fed-NGA-style aggregation: average normalized client directions."""
    norms = [update_norm(u) for u in updates]
    valid_norms = [n for n in norms if n > 0]
    if not valid_norms:
        return weighted_average(updates, counts), {"selected_clients": list(range(len(updates))), "method_core": "Fed-NGA normalized gradients fallback"}
    target_norm = float(np.median(valid_norms))
    normalized = []
    for update, norm_v in zip(updates, norms):
        denom = max(norm_v, 1e-12)
        normalized.append({k: v / denom for k, v in update.items()})
    avg_dir = weighted_average(normalized, counts)
    dir_norm = update_norm(avg_dir)
    if dir_norm > 0:
        avg_dir = {k: v * (target_norm / dir_norm) for k, v in avg_dir.items()}
    return avg_dir, {"selected_clients": list(range(len(updates))), "method_core": "Fed-NGA normalized update aggregation", "client_norms": norms, "target_norm": target_norm}


def huber_brfl_aggregate(updates: List[Dict[str, torch.Tensor]], config: ExperimentConfig) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """AAAI'24 Huber-loss robust aggregation core reproduction."""
    vecs = torch.stack([vectorize(u) for u in updates], dim=0)
    theta = torch.median(vecs, dim=0).values
    final_weights = torch.ones(vecs.shape[0], device=vecs.device)
    cutoffs: List[float] = []
    for _ in range(12):
        residuals = torch.linalg.vector_norm(vecs - theta.unsqueeze(0), dim=1)
        residual_np = residuals.detach().cpu().numpy().astype(float).tolist()
        med, scale = robust_median_mad(residual_np)
        cutoff = max(med + 1.345 * scale, 1e-8)
        cutoffs.append(float(cutoff))
        weights = torch.clamp(torch.tensor(cutoff, device=vecs.device) / torch.clamp(residuals, min=1e-12), max=1.0)
        if float(weights.sum().detach().cpu()) <= 0:
            break
        new_theta = (weights.unsqueeze(1) * vecs).sum(dim=0) / weights.sum()
        if float(torch.linalg.vector_norm(new_theta - theta).detach().cpu()) < 1e-8:
            theta = new_theta
            final_weights = weights
            break
        theta = new_theta
        final_weights = weights
    selected = [int(i) for i, w in enumerate(final_weights.detach().cpu().numpy().astype(float).tolist()) if w >= 0.5]
    return devectorize(theta, updates[0]), {"selected_clients": selected, "method_core": "Huber-BRFL iterative Huber M-estimator", "huber_weights": [float(x) for x in final_weights.detach().cpu().numpy().tolist()], "huber_cutoffs": cutoffs}


def adaaggrl_aggregate(updates: List[Dict[str, torch.Tensor]], server_update: Dict[str, torch.Tensor], config: ExperimentConfig) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """AdaAggRL-inspired adaptive aggregation core, not an official-code copy."""
    center = median_update(updates)
    distances = [update_distance(u, center) for u in updates]
    d_med, d_scale = robust_median_mad(distances)
    server_scores = []
    center_scores = []
    for update in updates:
        sc = cosine_between(server_update, update)
        cc = cosine_between(center, update)
        server_scores.append(max(0.0, 0.0 if math.isnan(sc) else sc))
        center_scores.append(max(0.0, 0.0 if math.isnan(cc) else cc))
    stability = [math.exp(-max(0.0, d - d_med) / d_scale) for d in distances]
    scores = [0.45 * sv + 0.35 * cv + 0.20 * st for sv, cv, st in zip(server_scores, center_scores, stability)]
    min_keep = max(1, len(updates) - config.num_malicious)
    selected = select_top_scores(scores, keep_ratio=min_keep / max(len(updates), 1), min_keep=min_keep)
    weights = softmax_weights([scores[i] for i in selected], config.act_temperature)
    return weighted_average([updates[i] for i in selected], weights), {"selected_clients": selected, "method_core": "AdaAggRL-core adaptive stability weighting", "stability_scores": stability, "server_scores": server_scores, "center_scores": center_scores, "policy_scores": scores}


def fedamm_aggregate(updates: List[Dict[str, torch.Tensor]], server_update: Dict[str, torch.Tensor], config: ExperimentConfig) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """FedAMM-inspired PCA/critical-parameter clustering core."""
    vecs_t = torch.stack([vectorize(u) for u in updates], dim=0)
    vecs = vecs_t.detach().cpu().numpy().astype(float)
    n = vecs.shape[0]
    if n <= 2:
        return weighted_average(updates, [1.0] * n), {"selected_clients": list(range(n)), "method_core": "FedAMM PCA cluster fallback"}
    centered = vecs - vecs.mean(axis=0, keepdims=True)
    vt = None
    try:
        _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
        k = max(1, min(5, n - 1, vt.shape[0]))
        proj = centered @ vt[:k].T
    except np.linalg.LinAlgError:
        proj = centered[:, :max(1, min(5, centered.shape[1]))]
    labels = KMeans(n_clusters=2, random_state=config.seed, n_init=10).fit_predict(proj)
    if vt is not None:
        server_vec = vectorize(server_update).detach().cpu().numpy().astype(float)
        server_proj = (server_vec.reshape(1, -1) - vecs.mean(axis=0, keepdims=True)) @ vt[:proj.shape[1]].T
    else:
        server_proj = np.zeros((1, proj.shape[1]))
    cluster_scores = []
    for label in [0, 1]:
        idx = np.where(labels == label)[0]
        if len(idx) == 0:
            cluster_scores.append(float("inf"))
        else:
            centroid = proj[idx].mean(axis=0, keepdims=True)
            cluster_scores.append(float(np.linalg.norm(centroid - server_proj)))
    keep_label = int(np.argmin(cluster_scores))
    selected = [int(i) for i, lab in enumerate(labels) if int(lab) == keep_label]
    min_keep = max(1, n - config.num_malicious)
    if len(selected) < min_keep:
        center = median_update(updates)
        distances = [update_distance(u, center) for u in updates]
        selected = [int(i) for i in np.argsort(np.asarray(distances))[:min_keep]]
    return weighted_average([updates[i] for i in selected], [1.0] * len(selected)), {"selected_clients": selected, "method_core": "FedAMM-core PCA critical-parameter clustering", "cluster_labels": [int(x) for x in labels.tolist()], "cluster_scores": cluster_scores}


def fedaa_aggregate(updates: List[Dict[str, torch.Tensor]], server_update: Dict[str, torch.Tensor], fairness_details: Optional[List[Dict[str, float]]], config: ExperimentConfig) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """FedAA-core reproduction: proximity client selection plus reward-weighted adaptive aggregation."""
    n = len(updates)
    vecs = [vectorize(u) for u in updates]
    stacked = torch.stack(vecs, dim=0)
    pairwise = torch.cdist(stacked, stacked, p=2)
    distance_sums = pairwise.sum(dim=1).detach().cpu().numpy().astype(float).tolist()
    keep = max(1, n - config.num_malicious)
    selected = [int(i) for i in np.argsort(np.asarray(distance_sums))[:keep]]
    reward_scores = []
    clean_accs = []
    fair_risks = []
    alignments = []
    for i in range(n):
        detail = fairness_details[i] if fairness_details is not None and i < len(fairness_details) else {}
        acc = float(detail.get("accuracy", 0.0)) if math.isfinite(float(detail.get("accuracy", 0.0))) else 0.0
        aeod = float(detail.get("aeod", 1.0)) if math.isfinite(float(detail.get("aeod", 1.0))) else 1.0
        aspd = float(detail.get("aspd", 1.0)) if math.isfinite(float(detail.get("aspd", 1.0))) else 1.0
        fair = max(aeod, aspd)
        cos = cosine_between(server_update, updates[i])
        align = max(0.0, 0.0 if math.isnan(cos) else cos)
        clean_accs.append(acc); fair_risks.append(fair); alignments.append(align)
    acc_z = normalize_scores(clean_accs, higher_is_better=True, clip=5.0)
    fair_z = normalize_scores(fair_risks, higher_is_better=False, clip=5.0)
    align_z = normalize_scores(alignments, higher_is_better=True, clip=5.0)
    for i in range(n):
        reward_scores.append(float(acc_z[i] + 0.75 * fair_z[i] + 0.25 * align_z[i]))
    weights = softmax_weights([reward_scores[i] for i in selected], max(config.act_temperature, 0.05))
    return weighted_average([updates[i] for i in selected], weights), {
        "selected_clients": selected,
        "method_core": "FedAA-core proximity selection with clean-root reward weights",
        "distance_sums": distance_sums,
        "reward_scores": reward_scores,
        "clean_utility": clean_accs,
        "fairness_risks": fair_risks,
        "alignment_scores": alignments,
    }


def act_fairness_risks(fairness: List[float], fairness_details: Optional[List[Dict[str, float]]], config: ExperimentConfig) -> Tuple[List[float], Dict[str, Any]]:
    mode = config.act_fairness_metric
    risks: List[float] = []
    aeods: List[float] = []
    aspds: List[float] = []
    for idx, fair in enumerate(fairness):
        detail = fairness_details[idx] if fairness_details is not None and idx < len(fairness_details) else {}
        aeod = detail.get("aeod", fair)
        aspd = detail.get("aspd", fair)
        aeod = float(aeod) if not math.isnan(float(aeod)) else 1.0
        aspd = float(aspd) if not math.isnan(float(aspd)) else 1.0
        aeods.append(aeod)
        aspds.append(aspd)
        if mode == "aeod":
            risks.append(aeod)
        elif mode == "aspd":
            risks.append(aspd)
        elif mode == "max":
            risks.append(max(aeod, aspd))
        else:
            risks.append(0.5 * (aeod + aspd))
    return risks, {"act_fairness_metric": mode, "root_aeod_values": aeods, "root_aspd_values": aspds}


def normalize_scores(values: Sequence[float], higher_is_better: bool = True, clip: Optional[float] = None) -> List[float]:
    clean = [float(v) if math.isfinite(float(v)) else 0.0 for v in values]
    med, scale = robust_median_mad(clean)
    signed = [(v - med) / scale for v in clean]
    if not higher_is_better:
        signed = [-v for v in signed]
    if clip is not None and clip > 0:
        signed = [max(-float(clip), min(float(clip), v)) for v in signed]
    return [float(v) for v in signed]


def guardfed_act_aggregate(updates: List[Dict[str, torch.Tensor]], fairness: List[float], server_update: Dict[str, torch.Tensor], config: ExperimentConfig, fairness_details: Optional[List[Dict[str, float]]] = None) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """Adaptive constrained trust aggregation over client updates only.

    The score is a primal-dual surrogate for maximizing clean-server utility
    under a fairness-risk budget, with robust centrality and root alignment as
    attack-resistance regularizers. It does not select among external baseline
    aggregators.
    """
    n = len(updates)
    risks, fair_info = act_fairness_risks(fairness, fairness_details, config)
    clean_accs = []
    for idx in range(n):
        detail = fairness_details[idx] if fairness_details is not None and idx < len(fairness_details) else {}
        acc = detail.get("accuracy", 0.0)
        clean_accs.append(float(acc) if math.isfinite(float(acc)) else 0.0)

    center = median_update(updates)
    distances = [update_distance(update, center) for update in updates]
    centrality_raw = [-d for d in distances]
    alignments = []
    for update in updates:
        cos = cosine_between(server_update, update)
        alignments.append(max(0.0, 0.0 if math.isnan(cos) else cos))

    violations = [max(0.0, risk - config.act_fairness_budget) for risk in risks]
    mean_risk = float(np.mean(risks)) if risks else 1.0
    dual_lambda = math.log1p(math.exp((mean_risk - config.act_fairness_budget) / max(config.act_temperature, 1e-6)))

    clip = config.ad2_score_clip
    utility_z = normalize_scores(clean_accs, higher_is_better=True, clip=clip)
    centrality_z = normalize_scores(centrality_raw, higher_is_better=True, clip=clip)
    alignment_z = normalize_scores(alignments, higher_is_better=True, clip=clip)
    violation_z = normalize_scores(violations, higher_is_better=False, clip=clip)
    risk_z = normalize_scores(risks, higher_is_better=False, clip=clip)

    # Constrained additive objective: learnable-style weights over utility, robustness, and fairness risk.
    scores = []
    component_contributions = []
    for u, c, a, rv, rz in zip(utility_z, centrality_z, alignment_z, violation_z, risk_z):
        terms = {
            "U": config.ad2_utility_weight * u,
            "C": config.ad2_centrality_weight * c,
            "A": config.ad2_alignment_weight * a,
            "F": config.act_risk_weight * rz,
            "V": config.act_violation_weight * dual_lambda * rv,
        }
        # Apply after all candidate overrides: a deleted term cannot be re-enabled.
        if config.ablation_component in terms:
            terms[config.ablation_component] = 0.0
        fairness_term = terms["F"] + terms["V"]
        robustness_term = terms["C"] + terms["A"]
        utility_term = terms["U"]
        scores.append(float(utility_term + robustness_term + fairness_term))
        component_contributions.append(terms)

    dist_med, dist_scale = robust_median_mad(distances)
    hard_gate = [i for i, (d, a) in enumerate(zip(distances, alignments)) if d <= dist_med + 2.5 * dist_scale or a > 0.0]
    if len(hard_gate) < max(1, n - config.num_malicious):
        hard_gate = list(range(n))
    gated_scores = [scores[i] if i in hard_gate else -1e9 for i in range(n)]
    selected = select_top_scores(gated_scores, min(max(config.act_keep_ratio, 0.05), 1.0), min_keep=max(1, n - config.num_malicious))

    selected_norms = [update_norm(updates[i]) for i in selected]
    norm_med, norm_scale = robust_median_mad(selected_norms)
    server_norm = update_norm(server_update)
    clip_cap = max(server_norm, norm_med + config.ad2_norm_clip_scale * norm_scale) if selected_norms else server_norm
    scaled = []
    norm_clip_scales = []
    for i in selected:
        norm = update_norm(updates[i])
        if config.ablation_component == "N":
            scale = 1.0
        elif config.ad2_norm_mode == "root":
            scale = server_norm / (norm + 1e-12) if server_norm > 0 and norm > 0 else 1.0
        else:
            scale = min(1.0, clip_cap / (norm + 1e-12)) if norm > 0 and clip_cap > 0 else 1.0
        norm_clip_scales.append(float(scale))
        scaled.append({k: v * scale for k, v in updates[i].items()})
    weights = softmax_weights([scores[i] for i in selected], config.act_temperature)
    client_weights = [0.0] * n
    for i, weight in zip(selected, weights):
        client_weights[i] = float(weight / sum(weights))
    return weighted_average(scaled, weights), {
        "ablation_component": config.ablation_component,
        "component_contributions": component_contributions,
        "component_raw": {"U": clean_accs, "C": centrality_raw, "A": alignments, "F": risks, "V": violations},
        "component_standardized": {"U": utility_z, "C": centrality_z, "A": alignment_z, "F": risk_z, "V": violation_z},
        "client_weights": client_weights,
        "aggregation_temperature": config.act_temperature,
        "selected_clients": selected,
        "method_core": "AD2 bounded dual-objective trust with adaptive/root norm scaling",
        "norm_mode": config.ad2_norm_mode,
        "score_clip": config.ad2_score_clip,
        "utility_weight": config.ad2_utility_weight,
        "centrality_weight": config.ad2_centrality_weight,
        "alignment_weight": config.ad2_alignment_weight,
        "norm_clip_cap": clip_cap,
        "norm_clip_scales": norm_clip_scales,
        "selected_update_norms": selected_norms,
        "trust_scores": scores,
        "clean_utility": clean_accs,
        "utility_z": utility_z,
        "centrality_z": centrality_z,
        "alignment_z": alignment_z,
        "risk_z": risk_z,
        "violation_z": violation_z,
        "fairness_risks": risks,
        "dual_lambda": dual_lambda,
        "mean_fairness": mean_risk,
        "fairness_budget": config.act_fairness_budget,
        "hard_gate_clients": hard_gate,
        **fair_info,
    }



def clone_config_with(config: ExperimentConfig, overrides: Dict[str, Any]) -> ExperimentConfig:
    new_config = copy.copy(config)
    for key, value in overrides.items():
        setattr(new_config, key, value)
    return new_config


def ad2plus_candidate_overrides(config: ExperimentConfig) -> List[Dict[str, Any]]:
    """Internal AD2+ candidate family used by the clean-root selector.

    The candidates are not external methods. They are different lenses of the
    same AD2 additive utility-robustness-fairness objective: balanced fairness,
    strict dual-risk control, ASPD-oriented, AEOD-oriented, and utility-stable.
    """
    base = {
        "ad2_score_clip": 0.0,
        "ad2_norm_mode": "root",
        "ad2_calibration_objective": "original",
        "ad2_norm_clip_scale": config.ad2_norm_clip_scale,
        "ad2_calibration_max_acc_drop": config.ad2_calibration_max_acc_drop,
    }
    candidates = [
        ("balanced", {"act_fairness_metric": "aeod_aspd", "act_risk_weight": 0.75, "act_violation_weight": 0.20, "act_keep_ratio": 0.80, "act_temperature": 0.35, "ad2_utility_weight": 1.00, "ad2_centrality_weight": 0.35, "ad2_alignment_weight": 0.35}),
        ("balanced_open", {"act_fairness_metric": "aeod_aspd", "act_risk_weight": 0.75, "act_violation_weight": 0.20, "act_keep_ratio": 0.90, "act_temperature": 0.35, "ad2_utility_weight": 1.00, "ad2_centrality_weight": 0.35, "ad2_alignment_weight": 0.35}),
        ("fair_stable", {"act_fairness_metric": "aeod_aspd", "act_risk_weight": 0.90, "act_violation_weight": 0.25, "act_keep_ratio": 0.80, "act_temperature": 0.35, "ad2_utility_weight": 1.00, "ad2_centrality_weight": 0.35, "ad2_alignment_weight": 0.35}),
        ("utility_fair", {"act_fairness_metric": "aeod_aspd", "act_risk_weight": 0.60, "act_violation_weight": 0.10, "act_keep_ratio": 0.80, "act_temperature": 0.35, "ad2_utility_weight": 1.20, "ad2_centrality_weight": 0.50, "ad2_alignment_weight": 0.50}),
        ("dual_strict", {"act_fairness_metric": "aeod_aspd", "act_risk_weight": 1.10, "act_violation_weight": 0.35, "act_keep_ratio": 0.80, "act_temperature": 0.35, "ad2_utility_weight": 0.70, "ad2_centrality_weight": 0.25, "ad2_alignment_weight": 0.25}),
        ("dual_sharp", {"act_fairness_metric": "aeod_aspd", "act_risk_weight": 1.30, "act_violation_weight": 0.50, "act_keep_ratio": 0.70, "act_temperature": 0.20, "ad2_utility_weight": 0.60, "ad2_centrality_weight": 0.20, "ad2_alignment_weight": 0.20}),
        ("aeod_focus", {"act_fairness_metric": "aeod", "act_risk_weight": 0.85, "act_violation_weight": 0.25, "act_keep_ratio": 0.80, "act_temperature": 0.35, "ad2_utility_weight": 0.80, "ad2_centrality_weight": 0.35, "ad2_alignment_weight": 0.35}),
        ("aspd_focus", {"act_fairness_metric": "aspd", "act_risk_weight": 0.85, "act_violation_weight": 0.25, "act_keep_ratio": 0.80, "act_temperature": 0.35, "ad2_utility_weight": 0.80, "ad2_centrality_weight": 0.35, "ad2_alignment_weight": 0.35}),
        ("aspd_strict", {"act_fairness_metric": "aspd", "act_risk_weight": 1.10, "act_violation_weight": 0.35, "act_keep_ratio": 0.70, "act_temperature": 0.20, "ad2_utility_weight": 0.70, "ad2_centrality_weight": 0.25, "ad2_alignment_weight": 0.25}),
        ("max_guard", {"act_fairness_metric": "max", "act_risk_weight": 0.85, "act_violation_weight": 0.25, "act_keep_ratio": 0.80, "act_temperature": 0.35, "ad2_utility_weight": 0.80, "ad2_centrality_weight": 0.35, "ad2_alignment_weight": 0.35}),
    ]
    merged = []
    for name, overrides in candidates:
        item = dict(base)
        item.update(overrides)
        item["candidate_name"] = name
        merged.append(item)
    return merged


def finite_metric(value: Any, fallback: float) -> float:
    try:
        out = float(value)
    except Exception:
        return fallback
    return out if math.isfinite(out) else fallback


def ad2plus_root_loss(metrics: Dict[str, Any]) -> float:
    aeod = finite_metric(metrics.get("aeod"), 1.0)
    aspd = finite_metric(metrics.get("aspd"), 1.0)
    return 0.45 * aeod + 0.45 * aspd + 0.10 * max(aeod, aspd)


def guardfed_ad2plus_adaptive_aggregate(updates: List[Dict[str, torch.Tensor]], fairness: List[float], server_update: Dict[str, torch.Tensor], config: ExperimentConfig, fairness_details: Optional[List[Dict[str, float]]] = None, global_state: Optional[Dict[str, torch.Tensor]] = None, bundle: Optional[Dict[str, Any]] = None, device: Optional[torch.device] = None) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """AD2+ clean-root adaptive selector over AD2 internal scoring lenses.

    Each round builds several AD2 candidate updates and evaluates only their
    one-step clean-root behavior. The selected candidate minimizes a fairness
    loss under an adaptive accuracy floor, so the selector is data-driven but
    never uses test labels or external baseline outputs.
    """
    if global_state is None or bundle is None or device is None:
        update, info = guardfed_act_aggregate(updates, fairness, server_update, config, fairness_details=fairness_details)
        info["ad2_plus_mode"] = "fallback_fixed_missing_root_context"
        return update, info

    candidate_rows: List[Dict[str, Any]] = []
    for overrides in ad2plus_candidate_overrides(config):
        name = str(overrides.pop("candidate_name"))
        candidate_config = clone_config_with(config, overrides)
        candidate_update, candidate_info = guardfed_act_aggregate(updates, fairness, server_update, candidate_config, fairness_details=fairness_details)
        candidate_state = {k: global_state[k] + candidate_update[k] for k in global_state}
        root_metrics = evaluate_state_on_server(candidate_state, bundle["num_features"], bundle, candidate_config, device)
        candidate_rows.append({
            "name": name,
            "update": candidate_update,
            "info": candidate_info,
            "root_metrics": {k: finite_metric(root_metrics.get(k), 1.0 if k != "accuracy" else 0.0) for k in METRICS},
            "config": {k: overrides[k] for k in sorted(overrides)},
        })

    best_acc = max(row["root_metrics"]["accuracy"] for row in candidate_rows) if candidate_rows else 0.0
    # Accuracy-first gate: the clean-root selector may adapt fairness weights,
    # but it cannot choose a candidate that noticeably sacrifices utility.
    # This avoids the degenerate fairness-optimal / under-trained behavior that
    # can otherwise look good only because predictions collapse.
    strict_acc_drop = min(max(0.0, config.ad2_calibration_max_acc_drop), 0.005)
    acc_floor = best_acc - strict_acc_drop
    for row in candidate_rows:
        acc = row["root_metrics"]["accuracy"]
        fair_loss = ad2plus_root_loss(row["root_metrics"])
        acc_shortfall = max(0.0, acc_floor - acc)
        budget_shortfall = max(0.0, max(row["root_metrics"]["aeod"], row["root_metrics"]["aspd"]) - config.ad2_calibration_budget)
        row["root_fairness_loss"] = fair_loss
        row["root_selection_score"] = acc - 0.35 * fair_loss - 6.00 * acc_shortfall - 0.10 * budget_shortfall
        row["root_acc_floor"] = acc_floor

    feasible_rows = [row for row in candidate_rows if row["root_metrics"]["accuracy"] >= acc_floor]
    selection_pool = feasible_rows or candidate_rows
    selected_row = max(selection_pool, key=lambda row: (row["root_selection_score"], -row["root_fairness_loss"], row["root_metrics"]["accuracy"]))
    info = dict(selected_row["info"])
    info.update({
        "method_core": "AD2+ clean-root adaptive dual-objective selector",
        "ad2_plus_mode": "adaptive_internal_candidate_selector",
        "ad2_plus_selected_candidate": selected_row["name"],
        "ad2_plus_selected_root_metrics": selected_row["root_metrics"],
        "ad2_plus_selected_root_score": selected_row["root_selection_score"],
        "ad2_plus_acc_floor": selected_row["root_acc_floor"],
        "ad2_plus_candidates": [
            {
                "name": row["name"],
                "root_metrics": row["root_metrics"],
                "root_fairness_loss": row["root_fairness_loss"],
                "root_selection_score": row["root_selection_score"],
                "config": row["config"],
            }
            for row in candidate_rows
        ],
    })
    return selected_row["update"], info


def fltrust_aggregate(updates: List[Dict[str, torch.Tensor]], server_update: Dict[str, torch.Tensor], selected: Optional[List[int]]=None, fairness: Optional[List[float]]=None, fairness_lambda: float=0.0) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    if selected is None: selected = list(range(len(updates)))
    root_norm = update_norm(server_update); scores=[]; scaled=[]
    for idx in selected:
        upd = updates[idx]; cos = cosine_between(server_update, upd); trust = max(0.0, 0.0 if math.isnan(cos) else cos)
        if fairness is not None and fairness_lambda > 0:
            f = fairness[idx] if not math.isnan(float(fairness[idx])) else 1.0; trust *= math.exp(-fairness_lambda * float(f))
        norm = update_norm(upd); scale = root_norm / (norm + 1e-12) if root_norm > 0 and norm > 0 else 1.0
        scaled.append({k: v * scale for k, v in upd.items()}); scores.append(trust)
    if not scaled: scaled = updates; scores = [1.0] * len(updates); selected = list(range(len(updates)))
    if sum(scores) <= 0: scores = [1.0] * len(scaled)
    return weighted_average(scaled, scores), {"selected_clients": selected, "trust_scores": scores}

def aggregate_round(method: str, updates: List[Dict[str, torch.Tensor]], counts: List[int], fairness: List[float], server_update: Dict[str, torch.Tensor], config: ExperimentConfig, fairness_details: Optional[List[Dict[str, float]]] = None, global_state: Optional[Dict[str, torch.Tensor]] = None, bundle: Optional[Dict[str, Any]] = None, device: Optional[torch.device] = None) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    if method == "FedAvg":
        weights = [1.0] * len(updates) if config.aggregation_weighting == "equal" else counts
        return weighted_average(updates, weights), {"selected_clients": list(range(len(updates))), "aggregation_weighting": config.aggregation_weighting}
    if method == "Median": return median_update(updates), {"selected_clients": list(range(len(updates)))}
    if method == "FairFed":
        weights = [math.exp(-config.fairfed_beta * (f if not math.isnan(float(f)) else 1.0)) for f in fairness]
        return weighted_average(updates, weights), {"selected_clients": list(range(len(updates))), "fairness_weights": weights}
    if method == "FLTrust": return fltrust_aggregate(updates, server_update)
    if method == "FLGMM": return flgmm_aggregate(updates, counts)
    if method == "FLAURA": return flaura_aggregate(updates, server_update, config)
    if method == "LayerGuard": return layerguard_aggregate(updates, config)
    if method == "SmartFL": return smartfl_aggregate(updates, config)
    if method == "FLTG": return fltg_aggregate(updates, server_update, config)
    if method == "FedDNA": return feddna_aggregate(updates, server_update, config)
    if method == "LASA": return lasa_aggregate(updates, config)
    if method == "Fed-NGA": return fed_nga_aggregate(updates, counts)
    if method == "Huber-BRFL": return huber_brfl_aggregate(updates, config)
    if method == "LoGoFair":
        weights = [1.0] * len(updates) if config.aggregation_weighting == "equal" else counts
        return weighted_average(updates, weights), {"selected_clients": list(range(len(updates))), "method_core": "LoGoFair-style post-processing over FedAvg", "aggregation_weighting": config.aggregation_weighting}
    if method == "AdaAggRL": return adaaggrl_aggregate(updates, server_update, config)
    if method == "FedAMM": return fedamm_aggregate(updates, server_update, config)
    if method == "FedAA": return fedaa_aggregate(updates, server_update, fairness_details, config)
    if method == "FairGuard":
        if config.fairguard_mode == "none":
            selected = list(range(len(updates)))
        else:
            selected = fairguard_select(fairness, config.seed)
        weights = [1.0] * len(selected) if config.aggregation_weighting == "equal" else [counts[i] for i in selected]
        return weighted_average([updates[i] for i in selected], weights), {"selected_clients": selected, "aggregation_weighting": config.aggregation_weighting, "fairguard_mode": config.fairguard_mode}
    if method == "FLTrust+FairGuard":
        selected = None if config.fairguard_mode == "none" else fairguard_select(fairness, config.seed)
        update, info = fltrust_aggregate(updates, server_update, selected=selected)
        info["fairguard_mode"] = config.fairguard_mode
        return update, info
    if method == "GuardFed":
        trusts = []
        for update, fair in zip(updates, fairness):
            cos = cosine_between(server_update, update)
            dev = max(0.0, 0.0 if math.isnan(cos) else cos)
            f = fair if not math.isnan(float(fair)) else 1.0
            trusts.append(dev * math.exp(-config.guardfed_fairness_lambda * float(f)))
        selected = [i for i, score in enumerate(trusts) if score > config.trust_threshold]
        if not selected:
            selected = [int(np.argmax(trusts))]
        return weighted_average([updates[i] for i in selected], [1.0] * len(selected)), {"selected_clients": selected, "trust_scores": trusts}
    if method == "GuardFed-AD2+":
        if config.ad2_plus_mode == "fixed":
            return guardfed_act_aggregate(updates, fairness, server_update, config, fairness_details=fairness_details)
        return guardfed_ad2plus_adaptive_aggregate(updates, fairness, server_update, config, fairness_details=fairness_details, global_state=global_state, bundle=bundle, device=device)
    if method in {"GuardFed-ACT", "GuardFed-AD2"}:
        return guardfed_act_aggregate(updates, fairness, server_update, config, fairness_details=fairness_details)
    raise ValueError(f"Unknown method {method}")

def apply_foe_if_needed(local_state: Dict[str, torch.Tensor], global_state: Dict[str, torch.Tensor], client: Dict[str, Any], audit: Dict[str, Any], server_update: Dict[str, torch.Tensor], config: ExperimentConfig) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    pre = state_delta(local_state, global_state)
    if "foe" not in client["attack_types"]:
        return pre, audit
    mode = client.get("foe_mode", "state")
    if mode == "state":
        attacked_state = {k: FOE_SCALE * v.detach().clone() for k, v in local_state.items()}
        post = state_delta(attacked_state, global_state)
        impl = "attack_acc_0.5_state: local state_dict multiplied by -0.5 before aggregation"
    elif mode == "delta":
        post = {k: FOE_SCALE * v.detach().clone() for k, v in pre.items()}
        impl = "attack_acc_0.5_delta: local update delta multiplied by -0.5 before aggregation"
    elif mode == "zero":
        post = {k: torch.zeros_like(v) for k, v in pre.items()}
        impl = "attack_acc_zero_delta: malicious update delta zeroed before aggregation"
    elif mode == "fedsa":
        pre_vec = vectorize(pre)
        server_vec = vectorize(server_update)
        pre_norm = float(torch.linalg.vector_norm(pre_vec).detach().cpu())
        server_norm = float(torch.linalg.vector_norm(server_vec).detach().cpu())
        if pre_norm <= 0 or server_norm <= 0:
            post_vec = -config.fedsa_gain * pre_vec
        else:
            clean_dir = server_vec / torch.clamp(torch.linalg.vector_norm(server_vec), min=1e-12)
            # FedSA-inspired sliding control: push the malicious update against
            # the clean server direction, while bounding the norm for stealth.
            post_vec = pre_vec - float(config.fedsa_gain) * pre_norm * clean_dir
            max_norm = max(pre_norm, float(config.fedsa_norm_ratio) * pre_norm)
            post_norm = torch.linalg.vector_norm(post_vec)
            if float(post_norm.detach().cpu()) > max_norm:
                post_vec = post_vec * (max_norm / torch.clamp(post_norm, min=1e-12))
        post = devectorize(post_vec, pre)
        impl = f"FedSA-inspired sliding performance attack: delta shifted against clean server update, gain={config.fedsa_gain}, norm_ratio_cap={config.fedsa_norm_ratio}"
    else:
        raise ValueError(f"Unsupported foe_mode: {mode}")
    audit["foe_impl"] = impl
    audit["foe_pre_update_norm"] = update_norm(pre)
    audit["foe_post_update_norm"] = update_norm(post)
    audit["foe_norm_ratio"] = audit["foe_post_update_norm"] / (audit["foe_pre_update_norm"] + 1e-12)
    audit["foe_pre_post_cosine"] = cosine_between(pre, post)
    return post, audit

def validate_attack_audit(attack: str, malicious_ids: Sequence[int], audits: List[Dict[str, Any]]) -> List[str]:
    errors=[]; byid = {a["client_id"]: a for a in audits}
    if attack == "Benign": return errors
    for cid in malicious_ids:
        a = byid.get(cid, {}); ats = a.get("attack_types", [])
        if "fflip" in ats:
            if a.get("fflip_requires_full_flip", True) and a.get("samples", 0) > 0 and not math.isclose(float(a.get("fflip_ratio", 0.0)), 1.0, abs_tol=1e-12): errors.append(f"client {cid} F Flip ratio {a.get('fflip_ratio')} != 1")
            if int(a.get("label_changed_count", 0)) != 0: errors.append(f"client {cid} F Flip changed labels")
        if "foe" in ats:
            ratio = float(a.get("foe_norm_ratio", 0.0)); cosine = float(a.get("foe_pre_post_cosine", 1.0))
            if ratio <= 1.05 and cosine > 0.90: errors.append(f"client {cid} FOE weak change: ratio={ratio}, cosine={cosine}")
    if attack == "Sp-DFA":
        groups = [tuple(byid.get(cid, {}).get("attack_types", [])) for cid in malicious_ids]
        if groups.count(("fflip",)) != len(malicious_ids)//2 or groups.count(("foe",)) != len(malicious_ids)//2: errors.append(f"Sp-DFA assignment wrong: {groups}")
    return errors


def create_client_data_dict(
    client_df: pd.DataFrame,
    feature_cols: List[str],
    label_col: str,
    sensitive_col: str,
    num_clients: int,
    alpha: float,
    device: torch.device,
    seed: int,
) -> Dict[int, Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    out: Dict[int, Dict[str, Any]] = {i: {"X": [], "y": [], "sensitive": []} for i in range(num_clients)}
    for group_val in [1, 0]:
        indices = client_df[client_df[sensitive_col] == group_val].index.to_numpy().copy()
        rng.shuffle(indices)
        cuts = (np.cumsum(rng.dirichlet([alpha] * num_clients)) * len(indices)).astype(int)[:-1]
        for cid, split in enumerate(np.split(indices, cuts)):
            if len(split) == 0:
                continue
            subset = client_df.loc[split]
            out[cid]["X"].append(torch.tensor(subset[feature_cols].values, dtype=torch.float32, device=device))
            out[cid]["y"].append(torch.tensor(subset[label_col].values, dtype=torch.long, device=device))
            out[cid]["sensitive"].append(subset[sensitive_col].values.astype(int))
    sensitive_feature_index = feature_cols.index(sensitive_col) if sensitive_col in feature_cols else None
    for cid in range(num_clients):
        if out[cid]["X"]:
            out[cid]["X"] = torch.cat(out[cid]["X"])
            out[cid]["y"] = torch.cat(out[cid]["y"])
            out[cid]["sensitive"] = np.concatenate(out[cid]["sensitive"])
        else:
            out[cid]["X"] = torch.empty(0, len(feature_cols), dtype=torch.float32, device=device)
            out[cid]["y"] = torch.empty(0, dtype=torch.long, device=device)
            out[cid]["sensitive"] = np.array([], dtype=int)
        out[cid]["sensitive_feature_index"] = sensitive_feature_index
    return out


def _adjust_counts_to_total(counts: Dict[Any, int], capacities: Dict[Any, int], total: int, rng: np.random.Generator) -> Dict[Any, int]:
    counts = {k: max(0, min(int(v), int(capacities.get(k, 0)))) for k, v in counts.items()}
    keys = list(counts)
    while sum(counts.values()) < total:
        available = [k for k in keys if counts[k] < capacities.get(k, 0)]
        if not available:
            break
        k = available[int(rng.integers(0, len(available)))]
        counts[k] += 1
    while sum(counts.values()) > total:
        positive = [k for k in keys if counts[k] > 0]
        if not positive:
            break
        k = positive[int(rng.integers(0, len(positive)))]
        counts[k] -= 1
    return counts


def sample_server_dataframe(
    train_df: pd.DataFrame,
    label_col: str,
    sensitive_col: str,
    config: ExperimentConfig,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    target = max(1, int(round(len(train_df) * config.server_ratio)))
    rng = np.random.default_rng(config.seed)
    mode = config.server_sampling
    if mode == "stratified_sensitive":
        parts = [group_df.sample(frac=config.server_ratio, random_state=config.seed) for _, group_df in train_df.groupby(sensitive_col)]
        server_df = pd.concat(parts).sort_index()
    elif mode == "iid_random":
        server_df = train_df.sample(n=target, replace=False, random_state=config.seed).sort_index()
    else:
        strata_cols = [sensitive_col, label_col]
        groups = {k: g for k, g in train_df.groupby(strata_cols, dropna=False)}
        keys = list(groups)
        capacities = {k: len(groups[k]) for k in keys}
        if mode == "group_balanced":
            raw_counts = {k: target // max(1, len(keys)) for k in keys}
        elif mode == "sensitive_balanced":
            raw_counts = {}
            for s_val, s_group in train_df.groupby(sensitive_col, dropna=False):
                s_target = target // max(1, train_df[sensitive_col].nunique())
                label_groups = {k: g for k, g in s_group.groupby(label_col, dropna=False)}
                for y_val, y_group in label_groups.items():
                    raw_counts[(s_val, y_val)] = int(round(s_target * len(y_group) / max(1, len(s_group))))
        elif mode == "class_balanced":
            raw_counts = {}
            for y_val, y_group in train_df.groupby(label_col, dropna=False):
                y_target = target // max(1, train_df[label_col].nunique())
                sens_groups = {k: g for k, g in y_group.groupby(sensitive_col, dropna=False)}
                for s_val, s_group in sens_groups.items():
                    raw_counts[(s_val, y_val)] = int(round(y_target * len(s_group) / max(1, len(y_group))))
        elif mode in {"dirichlet_strata", "dirichlet_strata_floor"}:
            alpha = float(config.server_alpha if config.server_alpha is not None else 5000.0)
            base = np.asarray([len(groups[k]) / len(train_df) for k in keys], dtype=float)
            concentration = np.maximum(base * alpha * len(keys), 1e-6)
            probs = rng.dirichlet(concentration)
            if mode == "dirichlet_strata_floor":
                min_per_group = max(5, int(round(0.02 * target)))
                floor_counts = {k: min(capacities[k], min_per_group) for k in keys}
                floor_total = sum(floor_counts.values())
                if floor_total >= target:
                    raw_counts = {k: int(round(target / max(1, len(keys)))) for k in keys}
                else:
                    remaining = target - floor_total
                    raw_counts = {k: floor_counts[k] + int(round(float(p) * remaining)) for k, p in zip(keys, probs)}
            else:
                raw_counts = {k: int(round(float(p) * target)) for k, p in zip(keys, probs)}
        elif mode in {"dirichlet_label_preserved", "dirichlet_label_preserved_strong_floor"}:
            # Preserve the label marginal distribution and apply Dirichlet skew
            # only over sensitive groups within each label. The strong-floor
            # variant keeps every sensitive-label stratum substantially
            # represented, so the alpha curve measures skew intensity instead
            # of root-group disappearance.
            alpha = float(config.server_alpha if config.server_alpha is not None else 5000.0)
            raw_counts = {}
            label_values = list(train_df[label_col].value_counts().sort_index().index)
            for y_val in label_values:
                y_group = train_df[train_df[label_col] == y_val]
                y_target = int(round(target * len(y_group) / len(train_df)))
                y_keys = [k for k in keys if k[1] == y_val]
                if not y_keys:
                    continue
                y_capacities = {k: capacities[k] for k in y_keys}
                base = np.asarray([capacities[k] / max(1, len(y_group)) for k in y_keys], dtype=float)
                concentration = np.maximum(base * alpha * len(y_keys), 1e-6)
                probs = rng.dirichlet(concentration)
                if mode == "dirichlet_label_preserved_strong_floor":
                    floor_fraction = 0.10
                    floor_min = 10
                else:
                    floor_fraction = 0.01
                    floor_min = 3
                min_per_group = min(max(floor_min, int(round(floor_fraction * y_target))), max(0, y_target // max(1, len(y_keys))))
                floor_counts = {k: min(y_capacities[k], min_per_group) for k in y_keys}
                floor_total = sum(floor_counts.values())
                remaining = max(0, y_target - floor_total)
                y_raw = {k: floor_counts[k] + int(round(float(p) * remaining)) for k, p in zip(y_keys, probs)}
                y_counts = _adjust_counts_to_total(y_raw, y_capacities, y_target, rng)
                raw_counts.update(y_counts)
        elif mode == "controlled_group_skew":
            # Deterministic sensitivity design for root/server distribution.
            # server_alpha is interpreted as skew strength rather than a
            # Dirichlet concentration: 0.0 keeps the global sensitive-label
            # group proportions, while values near 1.0 concentrate the clean
            # root set on one stratum. This makes the x-axis an auditable
            # distribution-shift level instead of a random Dirichlet draw.
            skew = float(config.server_alpha if config.server_alpha is not None else 0.0)
            skew = min(0.98, max(0.0, skew))
            base = np.asarray([capacities[k] / max(1, len(train_df)) for k in keys], dtype=float)
            dominant = max(keys, key=lambda k: capacities[k])
            one_hot = np.asarray([1.0 if k == dominant else 0.0 for k in keys], dtype=float)
            probs = (1.0 - skew) * base + skew * one_hot
            probs = probs / max(1e-12, probs.sum())
            raw_counts = {k: int(round(float(p) * target)) for k, p in zip(keys, probs)}
        elif mode == "controlled_positive_sensitive_skew":
            # Fairness-stress root/server distribution design. server_alpha is
            # again skew strength. Instead of drifting to the largest group, it
            # drifts toward the sensitive=1,label=1 stratum when available.
            # This produces an auditable root-data bias aligned with sensitive
            # and label attributes, making AEOD/ASPD sensitivity easier to
            # interpret than a random or majority-only skew.
            skew = float(config.server_alpha if config.server_alpha is not None else 0.0)
            skew = min(0.98, max(0.0, skew))
            base = np.asarray([capacities[k] / max(1, len(train_df)) for k in keys], dtype=float)
            target_key = (1, 1) if (1, 1) in capacities else max(keys, key=lambda k: (k[0], k[1], capacities[k]))
            one_hot = np.asarray([1.0 if k == target_key else 0.0 for k in keys], dtype=float)
            probs = (1.0 - skew) * base + skew * one_hot
            probs = probs / max(1e-12, probs.sum())
            raw_counts = {k: int(round(float(p) * target)) for k, p in zip(keys, probs)}
        elif mode == "controlled_target_group_skew":
            # Targeted fairness sensitivity search. server_alpha is skew
            # strength, while server_target_sensitive/server_target_label
            # choose the sensitive-label stratum to over-represent in the
            # clean root/server set. This lets us audit which root-data bias
            # actually degrades AEOD/ASPD without hand-editing results.
            skew = float(config.server_alpha if config.server_alpha is not None else 0.0)
            skew = min(0.98, max(0.0, skew))
            base = np.asarray([capacities[k] / max(1, len(train_df)) for k in keys], dtype=float)
            requested = (
                config.server_target_sensitive,
                config.server_target_label,
            )
            if requested[0] is not None and requested[1] is not None and requested in capacities:
                target_key = requested
            else:
                target_key = max(keys, key=lambda k: capacities[k])
            one_hot = np.asarray([1.0 if k == target_key else 0.0 for k in keys], dtype=float)
            probs = (1.0 - skew) * base + skew * one_hot
            probs = probs / max(1e-12, probs.sum())
            raw_counts = {k: int(round(float(p) * target)) for k, p in zip(keys, probs)}
        else:
            raise ValueError(f"Unsupported server_sampling: {mode}")
        counts = _adjust_counts_to_total(raw_counts, capacities, target, rng)
        parts = []
        for k, count in counts.items():
            if count > 0:
                parts.append(groups[k].sample(n=count, replace=False, random_state=int(rng.integers(0, 2**31 - 1))))
        server_df = pd.concat(parts).sort_index() if parts else train_df.sample(n=target, random_state=config.seed).sort_index()
    global_group_counts = train_df.groupby([sensitive_col, label_col]).size().sort_index().to_dict()
    server_group_counts_raw = server_df.groupby([sensitive_col, label_col]).size().sort_index().to_dict()
    all_group_keys = sorted(set(global_group_counts) | set(server_group_counts_raw))
    global_group_prob = {k: global_group_counts.get(k, 0) / max(1, len(train_df)) for k in all_group_keys}
    server_group_prob = {k: server_group_counts_raw.get(k, 0) / max(1, len(server_df)) for k in all_group_keys}
    group_tvd = 0.5 * sum(abs(server_group_prob[k] - global_group_prob[k]) for k in all_group_keys)
    group_kl = sum(
        server_group_prob[k] * math.log(max(server_group_prob[k], 1e-12) / max(global_group_prob[k], 1e-12))
        for k in all_group_keys
        if server_group_prob[k] > 0
    )
    max_group_abs_delta = max((abs(server_group_prob[k] - global_group_prob[k]) for k in all_group_keys), default=0.0)
    global_sensitive_prob = train_df[sensitive_col].value_counts(normalize=True).sort_index().to_dict()
    server_sensitive_prob = server_df[sensitive_col].value_counts(normalize=True).sort_index().to_dict()
    all_sensitive_keys = sorted(set(global_sensitive_prob) | set(server_sensitive_prob))
    sensitive_tvd = 0.5 * sum(abs(server_sensitive_prob.get(k, 0.0) - global_sensitive_prob.get(k, 0.0)) for k in all_sensitive_keys)
    global_label_prob = train_df[label_col].value_counts(normalize=True).sort_index().to_dict()
    server_label_prob = server_df[label_col].value_counts(normalize=True).sort_index().to_dict()
    all_label_keys = sorted(set(global_label_prob) | set(server_label_prob))
    label_tvd = 0.5 * sum(abs(server_label_prob.get(k, 0.0) - global_label_prob.get(k, 0.0)) for k in all_label_keys)
    audit = {
        "server_sampling": mode,
        "server_alpha": config.server_alpha,
        "server_target_sensitive": config.server_target_sensitive,
        "server_target_label": config.server_target_label,
        "server_rows": int(len(server_df)),
        "server_sensitive_counts": {str(k): int(v) for k, v in server_df[sensitive_col].value_counts().sort_index().to_dict().items()},
        "server_label_counts": {str(k): int(v) for k, v in server_df[label_col].value_counts().sort_index().to_dict().items()},
        "server_group_counts": {f"{int(k[0])}|{int(k[1])}": int(v) for k, v in server_group_counts_raw.items()},
        "global_group_counts": {f"{int(k[0])}|{int(k[1])}": int(v) for k, v in global_group_counts.items()},
        "server_group_prob": {f"{int(k[0])}|{int(k[1])}": float(server_group_prob[k]) for k in all_group_keys},
        "global_group_prob": {f"{int(k[0])}|{int(k[1])}": float(global_group_prob[k]) for k in all_group_keys},
        "group_tvd": float(group_tvd),
        "group_kl": float(group_kl),
        "max_group_abs_delta": float(max_group_abs_delta),
        "sensitive_tvd": float(sensitive_tvd),
        "label_tvd": float(label_tvd),
    }
    return server_df, audit


def apply_root_noise(root_df: pd.DataFrame, label_col: str, sensitive_col: str, config: ExperimentConfig):
    """Flip only root values using independent nested permutations; never use global RNG."""
    noisy = root_df.copy(deep=True)
    def supports(frame):
        return {f"{group}|{label}": int(((frame[sensitive_col] == group) & (frame[label_col] == label)).sum())
                for group in (0, 1) for label in (0, 1)}
    audit = {
        "root_rows": len(root_df),
        "clean_group_label_counts": supports(root_df),
        "clean_root_sha256": hashlib.sha256(pd.util.hash_pandas_object(root_df, index=False).values.tobytes()).hexdigest(),
        "mask_scheme": "local_SeedSequence(seed,240923,stream)_permutation_prefix_floor(rate*n)",
        "interpretation": "End-to-end root sensitivity: FedSA-inspired also uses the root update.",
    }
    for stream, (name, column, rate) in enumerate((
        ("label", label_col, config.root_label_noise),
        ("sensitive", sensitive_col, config.root_sensitive_noise),
    )):
        mask = np.zeros(len(root_df), dtype=bool)
        count = int(math.floor(rate * len(root_df)))
        if count:
            if not root_df[column].isin([0, 1]).all():
                raise ValueError(f"Root noise requires binary {column}")
            rng = np.random.default_rng(np.random.SeedSequence([config.seed, 240923, stream]))
            mask[rng.permutation(len(root_df))[:count]] = True
            noisy.loc[mask, column] = 1 - noisy.loc[mask, column]
        audit[name] = {"requested_rate": rate, "flipped_count": count,
                       "actual_rate": count / len(root_df) if len(root_df) else 0.0,
                       "mask_sha256": hashlib.sha256(mask.tobytes()).hexdigest()}
    audit["observed_group_label_counts"] = supports(noisy)
    return noisy, audit


def load_bundle(dataset: str, alpha: float, config: ExperimentConfig, device: torch.device) -> Dict[str, Any]:
    set_seed(config.seed, deterministic_image=(dataset == "celeba"))
    if dataset == "celeba":
        from src.celeba_data import load_celeba_bundle
        return load_celeba_bundle(alpha, config, sys.modules[__name__])
    loader_options = {"preprocessing_version": config.compas_preprocessing_version} if dataset == "compas" else {}
    loader = DatasetLoader(dataset_name=dataset, seed=config.seed, device=str(device), **loader_options)
    label_col = "income" if dataset == "adult" else "two_year_recid"
    feature_cols = [col for col in loader.train_df.columns if col != label_col and (config.include_sensitive_feature or col != loader.sensitive_column)]
    server_df, server_sampling_audit = sample_server_dataframe(loader.train_df, label_col, loader.sensitive_column, config)
    client_df = loader.train_df.drop(server_df.index).reset_index(drop=True)
    server_df = server_df.reset_index(drop=True)
    np.random.seed(config.seed)
    clients = create_client_data_dict(client_df, feature_cols, label_col, loader.sensitive_column, config.num_clients, alpha, device, config.seed)
    n_synth = int(round(len(loader.train_df) * config.synthetic_ratio))
    root_cols = list(dict.fromkeys(feature_cols + [loader.sensitive_column, label_col]))
    root_hash = int(pd.util.hash_pandas_object(server_df[root_cols], index=False).sum()) if len(server_df) else 0
    synth_key = (dataset, tuple(root_cols), config.synthetic_method, n_synth, config.seed, config.synthetic_epochs, root_hash)
    if n_synth > 0 and config.synthetic_method != "none" and synth_key in SYNTH_ROOT_CACHE:
        synth_df = SYNTH_ROOT_CACHE[synth_key].copy()
    else:
        synth_df = make_synthetic_root(config.synthetic_method, server_df, loader.train_df, feature_cols, label_col, loader.sensitive_column, n_synth, config.seed, config.synthetic_epochs)
        if n_synth > 0 and config.synthetic_method != "none":
            SYNTH_ROOT_CACHE[synth_key] = synth_df.copy()
    root_df = pd.concat([server_df[root_cols], synth_df], ignore_index=True)
    root_df, root_noise_audit = apply_root_noise(root_df, label_col, loader.sensitive_column, config)
    server_X = torch.tensor(root_df[feature_cols].values, dtype=torch.float32, device=device)
    server_y = torch.tensor(root_df[label_col].values, dtype=torch.long, device=device)
    server_sensitive = root_df[loader.sensitive_column].values.astype(int)
    X_test = torch.tensor(loader.test_df[feature_cols].values, dtype=torch.float32, device=device); y_test = torch.tensor(loader.test_df[label_col].values, dtype=torch.long, device=device); test_sensitive = np.asarray(loader.test_df[loader.sensitive_column].values, dtype=int)
    rw_weights = compute_reweighing_weights(loader.train_df, loader.sensitive_column, label_col) if config.use_reweighting else {(s, y): 1.0 for s in [0, 1] for y in [0, 1]}
    return {"dataset": dataset, "loader": loader, "label_col": label_col, "sensitive_col": loader.sensitive_column, "feature_cols": feature_cols, "feature_includes_sensitive": loader.sensitive_column in feature_cols, "feature_includes_label": label_col in feature_cols, "server_X": server_X, "server_y": server_y, "server_sensitive": server_sensitive, "root_clean_rows": int(len(server_df)), "root_synthetic_rows": int(len(synth_df)), "server_sampling_audit": server_sampling_audit, "root_noise_audit": root_noise_audit, "synthetic_method": config.synthetic_method, "clients": clients, "X_test": X_test, "y_test": y_test, "test_sensitive": test_sensitive, "num_features": len(feature_cols), "rw_weights": rw_weights, "train_rows": int(len(loader.train_df)), "test_rows": int(len(loader.test_df))}

def run_experiment(dataset: str, distribution: str, method: str, attack: str, config: ExperimentConfig, mode: str, device: torch.device, progress_callback=None, checkpoint_path=None) -> Dict[str, Any]:
    start = time.time(); alpha = config.client_alpha if config.client_alpha is not None else DISTRIBUTIONS[distribution]; bundle = load_bundle(dataset, alpha, config, device)
    if bundle["feature_includes_label"]: raise RuntimeError(f"Feature leakage detected for {dataset}: label column is in features")
    set_seed(config.seed, deterministic_image=(dataset == "celeba")); global_model = make_model(bundle, config, device); malicious_ids = list(range(config.num_malicious)); clients=[]; audits=[]
    for cid in range(config.num_clients):
        c, a = client_runtime_data(bundle["clients"][cid], cid, attack, malicious_ids, bundle["rw_weights"], device, config.fflip_mode, config.foe_mode, config.sdfa_foe_mode, config.spdfa_foe_mode); clients.append(c); audits.append(a)
    warnings=[]; round_summaries=[]; last10_metrics=[]; trajectory_metrics=[]
    for rnd in range(config.rounds):
        global_state = clone_state(global_model); server_update = train_server_update(global_model, bundle, config); updates=[]; counts=[]; fairness=[]; fairness_details=[]
        for client in clients:
            local_state = train_local_model(global_model, client, config); update, audits[client["cid"]] = apply_foe_if_needed(local_state, global_state, client, audits[client["cid"]], server_update, config); updates.append(update); counts.append(client["n"])
            state_for_eval = {k: global_state[k] + update[k] for k in global_state}; root_metrics = evaluate_state_on_server(state_for_eval, bundle["num_features"], bundle, config, device); fair = root_metrics["aeod"]; fairness.append(float(fair) if not math.isnan(float(fair)) else 1.0); fairness_details.append({k: float(root_metrics.get(k, math.nan)) for k in METRICS})
        if rnd == 0: warnings.extend(validate_attack_audit(attack, malicious_ids, audits))
        agg, info = aggregate_round(method, updates, counts, fairness, server_update, config, fairness_details=fairness_details, global_state=global_state, bundle=bundle, device=device); apply_update(global_model, agg)
        if config.full_round_diagnostics or rnd in {0, config.rounds - 1}:
            round_summaries.append({"round": rnd + 1, "aggregate": info, "client_ids": [c["cid"] for c in clients], "malicious_mask": [c["cid"] in malicious_ids for c in clients], "root_noise_audit": bundle.get("root_noise_audit"), "fairness_min": float(np.min(fairness)), "fairness_max": float(np.max(fairness)), "fairness_mean": float(np.mean(fairness))})
        round_metrics = evaluate_for_reporting(method, global_model, bundle, config)
        trajectory_metrics.append({"round": rnd + 1, "metrics": {k: round_metrics[k] for k in METRICS}})
        if progress_callback is not None:
            progress_callback(copy.deepcopy(trajectory_metrics[-1]))
        if rnd >= max(0, config.rounds - 10):
            round_metrics = evaluate_for_reporting(method, global_model, bundle, config)
            last10_metrics.append({"round": rnd + 1, "metrics": {k: round_metrics[k] for k in METRICS}})
    metrics = evaluate_for_reporting(method, global_model, bundle, config); warnings.extend(metrics.get("warnings", []))
    if any(w.startswith("client") or w.startswith("Sp-DFA") for w in warnings): raise RuntimeError("Attack self-check failed: " + "; ".join(warnings))
    if checkpoint_path is not None:
        torch.save(global_model.state_dict(), checkpoint_path)
    return {"run_id": make_run_id(mode, dataset, distribution, method, attack, config), "mode": mode, "dataset": dataset, "distribution": distribution, "alpha": alpha, "method": method, "attack": attack, "seed": config.seed, "rounds": config.rounds, "num_clients": config.num_clients, "num_malicious": config.num_malicious, "config": asdict(config), "metrics": {k: metrics[k] for k in METRICS}, "evaluation_stats": {k: metrics.get(k) for k in ("positive_rate", "majority_accuracy", "prediction_count")}, "warnings": warnings, "attack_audit": audits, "round_summaries": round_summaries, "last10_metrics": last10_metrics, "trajectory_metrics": trajectory_metrics, "data_contract": {"label_col": bundle["label_col"], "sensitive_col": bundle["sensitive_col"], "feature_includes_label": bundle["feature_includes_label"], "feature_includes_sensitive": bundle["feature_includes_sensitive"], "num_features": bundle["num_features"], "train_rows": bundle["train_rows"], "test_rows": bundle["test_rows"], "root_clean_rows": bundle.get("root_clean_rows"), "root_synthetic_rows": bundle.get("root_synthetic_rows"), "server_sampling": config.server_sampling, "server_alpha": config.server_alpha, "server_sampling_audit": bundle.get("server_sampling_audit"), "root_noise_audit": bundle.get("root_noise_audit"), "image_data_contract": bundle.get("image_data_contract"), "preprocessing_version": config.compas_preprocessing_version if dataset == "compas" else ("rgb64_v1" if dataset == "celeba" else "legacy"), "synthetic_method": bundle.get("synthetic_method")}, "attack_impl_note": f"F Flip mode={config.fflip_mode}; labels are unchanged. FOE mode={config.foe_mode}; S-DFA FOE mode={config.sdfa_foe_mode or config.foe_mode}; Sp-DFA FOE mode={config.spdfa_foe_mode or config.foe_mode}; FedSA mode uses foe_mode=fedsa.", "method_impl_note": f"All methods share identical preprocessing. sensitive_feature={config.include_sensitive_feature}, aggregation_weighting={config.aggregation_weighting}, use_reweighting={config.use_reweighting}, fairguard_mode={config.fairguard_mode}. FairGuard/GuardFed use root-data fairness filtering plus FLTrust-style cosine scoring; FLGMM/FLAURA/LayerGuard/SmartFL/FLTG/FedDNA/LASA/FedAA are core reproductions; GuardFed-AD2/AD2+ use adaptive dual-objective additive update weighting, norm scaling/clipping, and clean-server group-threshold calibration enabled={config.ad2_calibration_enabled}.", "duration_sec": time.time() - start}

def make_run_id(mode: str, dataset: str, distribution: str, method: str, attack: str, config: ExperimentConfig) -> str:
    revision_id = ""
    if dataset == "celeba":
        revision_id += f"|celeba=rgb64_v1|trainlimit={config.celeba_train_limit}|evallimit={config.celeba_eval_limit}|evalsplit={config.celeba_evaluation_split}"
    if config.root_label_noise or config.root_sensitive_noise:
        revision_id += f"|rootlabelnoise={config.root_label_noise}|rootsensitivenoise={config.root_sensitive_noise}"
    if config.compas_preprocessing_version != "legacy":
        revision_id += f"|compasprep={config.compas_preprocessing_version}"
    if config.ablation_component != "none" or config.client_alpha is not None or config.full_round_diagnostics:
        revision_id += f"|ablation={config.ablation_component}|clientalpha={config.client_alpha}|fullrounddiag={int(config.full_round_diagnostics)}"
    return "|".join([mode, dataset, distribution, method, attack, f"rounds={config.rounds}", f"seed={config.seed}", f"clients={config.num_clients}", f"malicious={config.num_malicious}", f"epochs={config.local_epochs}", f"batch={config.batch_size}", f"lr={config.learning_rate}", f"opt={config.optimizer}", f"root={config.server_ratio}+{config.synthetic_ratio}", f"serversamp={config.server_sampling}", f"serveralpha={config.server_alpha}", f"servertarget={config.server_target_sensitive}:{config.server_target_label}", f"synth={config.synthetic_method}", f"synth_epochs={config.synthetic_epochs}", f"sensfeat={int(config.include_sensitive_feature)}", f"agg={config.aggregation_weighting}", f"fflip={config.fflip_mode}", f"rw={int(config.use_reweighting)}", f"foe={config.foe_mode}", f"sdfafoe={config.sdfa_foe_mode or config.foe_mode}", f"spdfafoe={config.spdfa_foe_mode or config.foe_mode}", f"fedsa_gain={config.fedsa_gain}", f"fedsa_norm={config.fedsa_norm_ratio}", f"fg={config.fairguard_mode}", f"actfb={config.act_fairness_budget}", f"acttemp={config.act_temperature}", f"actkeep={config.act_keep_ratio}", f"actmetric={config.act_fairness_metric}", f"actdrop={config.act_anchor_drop}", f"actrisk={config.act_risk_weight}", f"actviol={config.act_violation_weight}", f"ad2calw={config.ad2_calibration_base_weight}", f"ad2calb={config.ad2_calibration_budget}", f"ad2calt={config.ad2_calibration_temperature}", f"ad2calq={config.ad2_calibration_quantiles}", f"ad2clip={config.ad2_score_clip}", f"ad2normclip={config.ad2_norm_clip_scale}", f"ad2accdrop={config.ad2_calibration_max_acc_drop}", f"ad2calobj={config.ad2_calibration_objective}", f"ad2calen={int(config.ad2_calibration_enabled)}", f"ad2normmode={config.ad2_norm_mode}", f"ad2uw={config.ad2_utility_weight}", f"ad2cw={config.ad2_centrality_weight}", f"ad2aw={config.ad2_alignment_weight}", f"ad2plus={config.ad2_plus_mode}", f"suite={config.experiment_suite}", f"tag={config.experiment_tag}"]) + revision_id

def load_completed(path: Path) -> set[str]:
    if not path.exists(): return set()
    done=set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try: done.add(json.loads(line)["run_id"])
            except Exception: pass
    return done

def append_result(path: Path, result: Dict[str, Any]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f: f.write(json.dumps(result, ensure_ascii=False) + "\n"); f.flush()

def make_jobs(args: argparse.Namespace) -> List[Tuple[str, str, str, str]]:
    datasets = args.datasets or DATASETS
    methods = args.methods or METHODS
    distributions = args.distributions or list(DISTRIBUTIONS.keys())
    attacks = args.attacks or ATTACKS
    jobs=[]
    if args.smoke:
        for dataset in datasets:
            for method in methods: jobs.append((dataset, "IID", method, "Benign"))
        for attack in ["F Flip", "FOE", "S-DFA", "Sp-DFA"]: jobs.append(("adult", "IID", "GuardFed", attack))
        return jobs
    for dataset in datasets:
        for dist in distributions:
            for method in methods:
                for attack in attacks: jobs.append((dataset, dist, method, attack))
    return jobs

def write_summary_tables(raw_path: Path = RAW_PATH) -> None:
    if not raw_path.exists(): return
    rows=[json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    full=[r for r in rows if r.get("mode") == "full"]; source=full if full else rows; latest={}
    for r in source: latest[(r["dataset"], r["method"], r["distribution"], r["attack"])] = r
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for dataset, path in TABLE_PATHS.items():
        cols=["method", "metric"] + [f"{dist}_{attack}" for dist in DISTRIBUTIONS for attack in ATTACKS]
        with path.open("w", newline="", encoding="utf-8") as f:
            w=csv.DictWriter(f, fieldnames=cols); w.writeheader()
            for method in METHODS:
                for metric in METRICS:
                    out={"method": method, "metric": metric}
                    for dist in DISTRIBUTIONS:
                        for attack in ATTACKS:
                            r=latest.get((dataset, method, dist, attack)); value=""
                            if r is not None:
                                value=r["metrics"].get(metric, "")
                                if metric == "accuracy" and value != "": value=float(value)*100.0
                            out[f"{dist}_{attack}"]=value
                    w.writerow(out)

def write_reproduction_report(results: List[Dict[str, Any]], mode: str, config: ExperimentConfig) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True); lines=["# GuardFed Table II/III Reproduction Report", "", f"Mode: {mode}", f"Completed cells in this invocation: {len(results)}", "", "## Protocol", "- Datasets: adult, compas", "- Distributions: IID alpha=5000, non-IID alpha=5", "- Methods: " + ", ".join(METHODS), "- Attacks: " + ", ".join(ATTACKS), "- Class-B FL: excluded", f"- Config: {asdict(config)}", f"- Sensitive attribute included in model inputs: {config.include_sensitive_feature}.", f"- Client aggregation weighting: {config.aggregation_weighting}.", f"- F Flip mode: {config.fflip_mode}; labels are unchanged.", f"- FOE mode: {config.foe_mode}; S-DFA FOE mode: {config.sdfa_foe_mode or config.foe_mode}; Sp-DFA FOE mode: {config.spdfa_foe_mode or config.foe_mode}.", f"- FairGuard mode: {config.fairguard_mode}.", f"- Reweighting enabled: {config.use_reweighting}.", "- Adult: label income (>50K=1), sensitive sex (Male=1, Female=0).", "- COMPAS: label two_year_recid, sensitive race (African-American=1, Others=0).", f"- Server/root data: {config.server_ratio:.1%} clean + {config.synthetic_ratio:.1%} Gaussian Copula synthetic.", "- FOE default: existing Git attack_acc_0.5 state_dict mode; S-DFA/Sp-DFA can override this and are recorded per run.", "", "## Outputs", f"- Raw JSONL: {RAW_PATH}", f"- Adult table: {TABLE_PATHS['adult']}", f"- COMPAS table: {TABLE_PATHS['compas']}", ""]
    if results:
        lines.append("## Latest Invocation Results")
        for r in results:
            m=r["metrics"]; warn=f" warnings={len(r.get('warnings', []))}" if r.get("warnings") else ""; lines.append(f"- {r['dataset']} {r['distribution']} {r['attack']} {r['method']}: ACC={m['accuracy']:.6f}, AEOD={m['aeod']:.6f}, ASPD={m['aspd']:.6f}, time={r['duration_sec']:.1f}s{warn}")
    else: lines.append("No new cells were run in this invocation; existing raw results were reused.")
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

def run_validation() -> None:
    proc = subprocess.run([sys.executable, str(ROOT / "scripts" / "validate_reproduction_data.py")], cwd=str(ROOT))
    if proc.returncode != 0: raise RuntimeError("Data/metric validation failed; full experiments are blocked")

def main() -> int:
    p=argparse.ArgumentParser(); g=p.add_mutually_exclusive_group(required=True); g.add_argument("--smoke", action="store_true"); g.add_argument("--full", action="store_true")
    p.add_argument("--datasets", nargs="*", choices=DATASETS)
    p.add_argument("--methods", nargs="*", choices=METHODS)
    p.add_argument("--distributions", nargs="*", choices=list(DISTRIBUTIONS.keys()))
    p.add_argument("--attacks", nargs="*", choices=ALL_ATTACKS)
    p.add_argument("--rounds", type=int)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--num-clients", type=int, default=20)
    p.add_argument("--num-malicious", type=int, default=4)
    p.add_argument("--local-epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--learning-rate", type=float, default=0.005)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu", "auto"])
    p.add_argument("--optimizer", default="adam", choices=["adam", "sgd"])
    p.add_argument("--server-ratio", type=float, default=0.05)
    p.add_argument("--synthetic-ratio", type=float, default=0.0)
    p.add_argument("--server-sampling", default="stratified_sensitive", choices=["stratified_sensitive", "iid_random", "dirichlet_strata", "dirichlet_strata_floor", "dirichlet_label_preserved", "dirichlet_label_preserved_strong_floor", "controlled_group_skew", "controlled_positive_sensitive_skew", "controlled_target_group_skew", "group_balanced", "sensitive_balanced", "class_balanced"])
    p.add_argument("--server-alpha", type=float)
    p.add_argument("--server-target-sensitive", type=int, choices=[0, 1])
    p.add_argument("--server-target-label", type=int, choices=[0, 1])
    p.add_argument("--synthetic-method", default="gaussian_copula", choices=["none", "gaussian_copula", "ctgan", "tvae", "bootstrap", "noisy_bootstrap", "smote", "pca_gaussian"])
    p.add_argument("--synthetic-epochs", type=int, default=50)
    p.add_argument("--include-sensitive-feature", action="store_true")
    p.add_argument("--aggregation-weighting", default="count", choices=["count", "equal"])
    p.add_argument("--fflip-mode", default="invert", choices=["invert", "label_conditioned", "label_conditioned_reverse", "all_privileged", "all_unprivileged"])
    p.add_argument("--foe-mode", default="state", choices=["state", "delta", "zero", "fedsa"])
    p.add_argument("--sdfa-foe-mode", choices=["state", "delta", "zero", "fedsa"])
    p.add_argument("--spdfa-foe-mode", choices=["state", "delta", "zero", "fedsa"])
    p.add_argument("--fedsa-gain", type=float, default=1.75)
    p.add_argument("--fedsa-norm-ratio", type=float, default=2.0)
    p.add_argument("--fairguard-mode", default="server_aeod", choices=["server_aeod", "none"])
    p.add_argument("--act-fairness-budget", type=float, default=0.06)
    p.add_argument("--act-temperature", type=float, default=0.35)
    p.add_argument("--act-keep-ratio", type=float, default=0.80)
    p.add_argument("--act-fairness-metric", default="aeod_aspd", choices=["aeod", "aspd", "aeod_aspd", "max"])
    p.add_argument("--act-anchor-drop", type=float, default=0.005)
    p.add_argument("--act-risk-weight", type=float, default=1.0)
    p.add_argument("--act-violation-weight", type=float, default=1.0)
    p.add_argument("--ad2-calibration-base-weight", type=float, default=1.0)
    p.add_argument("--ad2-calibration-budget", type=float, default=0.06)
    p.add_argument("--ad2-calibration-temperature", type=float, default=0.03)
    p.add_argument("--ad2-calibration-quantiles", type=int, default=41)
    p.add_argument("--ad2-score-clip", type=float, default=5.0)
    p.add_argument("--ad2-norm-clip-scale", type=float, default=2.5)
    p.add_argument("--ad2-calibration-max-acc-drop", type=float, default=0.03)
    p.add_argument("--ad2-calibration-objective", default="acc_floor", choices=["acc_floor", "original"])
    p.add_argument("--disable-ad2-calibration", action="store_true")
    p.add_argument("--ad2-norm-mode", default="adaptive", choices=["adaptive", "root"])
    p.add_argument("--ad2-utility-weight", type=float, default=1.0)
    p.add_argument("--ad2-centrality-weight", type=float, default=0.35)
    p.add_argument("--ad2-alignment-weight", type=float, default=0.35)
    p.add_argument("--ad2-plus-mode", default="adaptive", choices=["adaptive", "fixed"])
    p.add_argument("--experiment-suite", default="main")
    p.add_argument("--experiment-tag", default="default")
    p.add_argument("--no-reweighting", action="store_true")
    p.add_argument("--ablation-component", default="none", choices=["none", "U", "C", "A", "F", "V", "N"])
    p.add_argument("--client-alpha", type=float)
    p.add_argument("--full-round-diagnostics", action="store_true")
    p.add_argument("--root-label-noise", type=float, default=0.0)
    p.add_argument("--root-sensitive-noise", type=float, default=0.0)
    p.add_argument("--compas-preprocessing-version", choices=["legacy", "train_only"], default="legacy")
    p.add_argument("--force", action="store_true")
    args=p.parse_args(); rounds=args.rounds if args.rounds is not None else (1 if args.smoke else 70); config=ExperimentConfig(root_label_noise=args.root_label_noise, root_sensitive_noise=args.root_sensitive_noise, compas_preprocessing_version=args.compas_preprocessing_version, ablation_component=args.ablation_component, client_alpha=args.client_alpha, full_round_diagnostics=args.full_round_diagnostics, seed=args.seed, num_clients=args.num_clients, num_malicious=args.num_malicious, local_epochs=args.local_epochs, batch_size=args.batch_size, learning_rate=args.learning_rate, rounds=rounds, device=args.device, optimizer=args.optimizer, server_ratio=args.server_ratio, synthetic_ratio=args.synthetic_ratio, server_sampling=args.server_sampling, server_alpha=args.server_alpha, server_target_sensitive=args.server_target_sensitive, server_target_label=args.server_target_label, synthetic_method=args.synthetic_method, synthetic_epochs=args.synthetic_epochs, include_sensitive_feature=args.include_sensitive_feature, aggregation_weighting=args.aggregation_weighting, fflip_mode=args.fflip_mode, foe_mode=args.foe_mode, sdfa_foe_mode=args.sdfa_foe_mode, spdfa_foe_mode=args.spdfa_foe_mode, fedsa_gain=args.fedsa_gain, fedsa_norm_ratio=args.fedsa_norm_ratio, fairguard_mode=args.fairguard_mode, use_reweighting=not args.no_reweighting, act_fairness_budget=args.act_fairness_budget, act_temperature=args.act_temperature, act_keep_ratio=args.act_keep_ratio, act_fairness_metric=args.act_fairness_metric, act_anchor_drop=args.act_anchor_drop, act_risk_weight=args.act_risk_weight, act_violation_weight=args.act_violation_weight, ad2_calibration_base_weight=args.ad2_calibration_base_weight, ad2_calibration_budget=args.ad2_calibration_budget, ad2_calibration_temperature=args.ad2_calibration_temperature, ad2_calibration_quantiles=args.ad2_calibration_quantiles, ad2_score_clip=args.ad2_score_clip, ad2_norm_clip_scale=args.ad2_norm_clip_scale, ad2_calibration_max_acc_drop=args.ad2_calibration_max_acc_drop, ad2_calibration_objective=args.ad2_calibration_objective, ad2_calibration_enabled=not args.disable_ad2_calibration, ad2_norm_mode=args.ad2_norm_mode, ad2_utility_weight=args.ad2_utility_weight, ad2_centrality_weight=args.ad2_centrality_weight, ad2_alignment_weight=args.ad2_alignment_weight, ad2_plus_mode=args.ad2_plus_mode, experiment_suite=args.experiment_suite, experiment_tag=args.experiment_tag)
    run_validation(); device=choose_device(args.device); mode="smoke" if args.smoke else "full"; jobs=make_jobs(args); completed=load_completed(RAW_PATH); print(f"Running {len(jobs)} {mode} jobs on {device}"); new=[]
    for idx, (dataset, dist, method, attack) in enumerate(jobs, 1):
        run_id=make_run_id(mode, dataset, dist, method, attack, config)
        if run_id in completed and not args.force: print(f"[{idx}/{len(jobs)}] skip existing {run_id}"); continue
        print(f"[{idx}/{len(jobs)}] {dataset} {dist} {method} {attack}", flush=True); result=run_experiment(dataset, dist, method, attack, config, mode, device); append_result(RAW_PATH, result); completed.add(run_id); new.append(result); m=result["metrics"]; print(f"  -> ACC={m['accuracy']:.6f} AEOD={m['aeod']:.6f} ASPD={m['aspd']:.6f} time={result['duration_sec']:.1f}s", flush=True)
    write_summary_tables(RAW_PATH); write_reproduction_report(new, mode, config)
    if args.full: subprocess.run([sys.executable, str(ROOT / "scripts" / "compare_paper_tables.py")], cwd=str(ROOT), check=False)
    print(f"Done. Raw results: {RAW_PATH}"); return 0
if __name__ == "__main__": raise SystemExit(main())
