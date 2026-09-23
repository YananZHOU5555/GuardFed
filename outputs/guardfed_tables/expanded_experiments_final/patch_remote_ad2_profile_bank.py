from pathlib import Path

path = Path("/home/yannan/workspace/GuardFed/scripts/reproduce_paper_tables.py")
lines = path.read_text(encoding="utf-8").splitlines()

start = None
end = None
for i, line in enumerate(lines):
    if "balanced_b06_cal002_q81" in line:
        start = i
    if start is not None and "max_guard_b08_cal003_q81" in line:
        end = i
        break

if start is None or end is None:
    raise SystemExit("cross-budget profile lines not found")

replacement = [
    '        ("balanced_b06_cal002_q81", {"act_fairness_metric": "aeod_aspd", "act_risk_weight": 0.75, "act_violation_weight": 0.20, "act_keep_ratio": 0.80, "act_temperature": 0.35, "ad2_utility_weight": 1.00, "ad2_centrality_weight": 0.35, "ad2_alignment_weight": 0.35, "act_fairness_budget": 0.06, "ad2_calibration_budget": 0.06, "ad2_calibration_max_acc_drop": 0.02, "ad2_calibration_quantiles": 81}),',
    '        ("utility_fair_b06_cal002_q81", {"act_fairness_metric": "aeod_aspd", "act_risk_weight": 0.60, "act_violation_weight": 0.10, "act_keep_ratio": 0.80, "act_temperature": 0.35, "ad2_utility_weight": 1.20, "ad2_centrality_weight": 0.50, "ad2_alignment_weight": 0.50, "act_fairness_budget": 0.06, "ad2_calibration_budget": 0.06, "ad2_calibration_max_acc_drop": 0.02, "ad2_calibration_quantiles": 81}),',
    '        ("balanced_b08_cal003_q81", {"act_fairness_metric": "aeod_aspd", "act_risk_weight": 0.75, "act_violation_weight": 0.20, "act_keep_ratio": 0.80, "act_temperature": 0.35, "ad2_utility_weight": 1.00, "ad2_centrality_weight": 0.35, "ad2_alignment_weight": 0.35, "act_fairness_budget": 0.08, "ad2_calibration_budget": 0.08, "ad2_calibration_max_acc_drop": 0.03, "ad2_calibration_quantiles": 81}),',
    '        ("aspd_strict_b08_cal003_q81", {"act_fairness_metric": "aspd", "act_risk_weight": 1.10, "act_violation_weight": 0.35, "act_keep_ratio": 0.70, "act_temperature": 0.20, "ad2_utility_weight": 0.70, "ad2_centrality_weight": 0.25, "ad2_alignment_weight": 0.25, "act_fairness_budget": 0.08, "ad2_calibration_budget": 0.08, "ad2_calibration_max_acc_drop": 0.03, "ad2_calibration_quantiles": 81}),',
    '        ("max_guard_b08_cal003_q81", {"act_fairness_metric": "max", "act_risk_weight": 0.85, "act_violation_weight": 0.25, "act_keep_ratio": 0.80, "act_temperature": 0.35, "ad2_utility_weight": 0.80, "ad2_centrality_weight": 0.35, "ad2_alignment_weight": 0.35, "act_fairness_budget": 0.08, "ad2_calibration_budget": 0.08, "ad2_calibration_max_acc_drop": 0.03, "ad2_calibration_quantiles": 81}),',
]

lines[start : end + 1] = replacement
path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"fixed {path}:{start + 1}-{end + 1}")
