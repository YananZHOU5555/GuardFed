from pathlib import Path

path = Path("/home/yannan/workspace/GuardFed/scripts/reproduce_paper_tables.py")
text = path.read_text(encoding="utf-8")

old_helper = """def evaluate_state_on_server(state: Dict[str, torch.Tensor], input_size: int, bundle: Dict[str, Any], config: ExperimentConfig, device: torch.device) -> Dict[str, Any]:
    model = SimpleMLP(input_size, config.seed).to(device); model.load_state_dict(state)
    return evaluate_model(model, bundle["server_X"], bundle["server_y"], bundle["server_sensitive"], config.batch_size)
"""

new_helper = """def evaluate_state_on_server(state: Dict[str, torch.Tensor], input_size: int, bundle: Dict[str, Any], config: ExperimentConfig, device: torch.device) -> Dict[str, Any]:
    model = SimpleMLP(input_size, config.seed).to(device); model.load_state_dict(state)
    return evaluate_model(model, bundle["server_X"], bundle["server_y"], bundle["server_sensitive"], config.batch_size)

def evaluate_state_on_server_calibrated(state: Dict[str, torch.Tensor], input_size: int, bundle: Dict[str, Any], config: ExperimentConfig, device: torch.device) -> Dict[str, Any]:
    model = SimpleMLP(input_size, config.seed).to(device); model.load_state_dict(state)
    thresholds, info = fit_group_thresholds(model, bundle, config)
    metrics = dict(info.get("server_calibration_metrics", {}))
    metrics["calibration_thresholds"] = thresholds
    metrics["calibration_info"] = info
    return metrics
"""

if old_helper not in text:
    raise SystemExit("evaluate_state_on_server helper block not found")
text = text.replace(old_helper, new_helper, 1)

for name in [
    "balanced_b06_cal002_q81",
    "utility_fair_b06_cal002_q81",
    "balanced_b08_cal003_q81",
    "aspd_strict_b08_cal003_q81",
    "max_guard_b08_cal003_q81",
]:
    marker = f'("{name}", {{'
    idx = text.find(marker)
    if idx < 0:
        raise SystemExit(f"profile {name} not found")
    line_end = text.find("}),", idx)
    line = text[idx:line_end]
    if '"ad2_calibration_objective": "acc_floor"' not in line:
        text = text[:line_end] + ', "ad2_calibration_objective": "acc_floor"' + text[line_end:]

old_candidate_eval = """        candidate_state = {k: global_state[k] + candidate_update[k] for k in global_state}
        root_metrics = evaluate_state_on_server(candidate_state, bundle["num_features"], bundle, candidate_config, device)
        candidate_rows.append({
            "name": name,
            "update": candidate_update,
            "info": candidate_info,
            "root_metrics": {k: finite_metric(root_metrics.get(k), 1.0 if k != "accuracy" else 0.0) for k in METRICS},
            "config": {k: overrides[k] for k in sorted(overrides)},
        })
"""

new_candidate_eval = """        candidate_state = {k: global_state[k] + candidate_update[k] for k in global_state}
        root_metrics = evaluate_state_on_server_calibrated(candidate_state, bundle["num_features"], bundle, candidate_config, device)
        candidate_rows.append({
            "name": name,
            "update": candidate_update,
            "info": candidate_info,
            "root_metrics": {k: finite_metric(root_metrics.get(k), 1.0 if k != "accuracy" else 0.0) for k in METRICS},
            "root_metrics_source": "clean_root_calibrated",
            "config": {k: overrides[k] for k in sorted(overrides)},
        })
"""

if old_candidate_eval not in text:
    raise SystemExit("candidate eval block not found")
text = text.replace(old_candidate_eval, new_candidate_eval, 1)

old_selector = """    best_acc = max(row["root_metrics"]["accuracy"] for row in candidate_rows) if candidate_rows else 0.0
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
"""

new_selector = """    best_acc = max(row["root_metrics"]["accuracy"] for row in candidate_rows) if candidate_rows else 0.0
    # Constrained adaptive selector: first keep candidates within a clean-root
    # utility floor, then minimize calibrated fairness loss. This is closer to
    # the AD2+ method claim than a pure accuracy score, and it avoids treating
    # low-utility fairness collapse as a win.
    acc_tolerance = min(max(config.ad2_calibration_max_acc_drop, 0.005), 0.020)
    acc_floor = best_acc - acc_tolerance
    for row in candidate_rows:
        acc = row["root_metrics"]["accuracy"]
        fair_loss = ad2plus_root_loss(row["root_metrics"])
        acc_shortfall = max(0.0, acc_floor - acc)
        budget_shortfall = max(0.0, max(row["root_metrics"]["aeod"], row["root_metrics"]["aspd"]) - config.ad2_calibration_budget)
        row["root_fairness_loss"] = fair_loss
        row["root_selection_score"] = -fair_loss - 0.50 * budget_shortfall - 3.00 * acc_shortfall + 0.05 * max(0.0, acc - acc_floor)
        row["root_acc_floor"] = acc_floor
"""

if old_selector not in text:
    raise SystemExit("selector block not found")
text = text.replace(old_selector, new_selector, 1)

old_info = """                "root_fairness_loss": row["root_fairness_loss"],
                "root_selection_score": row["root_selection_score"],
                "config": row["config"],
"""

new_info = """                "root_fairness_loss": row["root_fairness_loss"],
                "root_selection_score": row["root_selection_score"],
                "root_metrics_source": row.get("root_metrics_source"),
                "config": row["config"],
"""
if old_info not in text:
    raise SystemExit("candidate info block not found")
text = text.replace(old_info, new_info, 1)

path.write_text(text, encoding="utf-8")
print("patched AD2+ selector v2")
