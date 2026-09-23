from pathlib import Path

path = Path('/home/yannan/workspace/GuardFed/scripts/reproduce_paper_tables.py')
text = path.read_text(encoding='utf-8-sig')

replacements = [
    (
        'warnings=[]; round_summaries=[]; last10_metrics=[]',
        'warnings=[]; round_summaries=[]; last10_metrics=[]; trajectory_metrics=[]',
    ),
    (
        '        if rnd >= max(0, config.rounds - 10):\n',
        '        round_metrics = evaluate_for_reporting(method, global_model, bundle, config)\n'
        '        trajectory_metrics.append({"round": rnd + 1, "metrics": {k: round_metrics[k] for k in METRICS}})\n'
        '        if rnd >= max(0, config.rounds - 10):\n',
    ),
    (
        '"last10_metrics": last10_metrics, "data_contract":',
        '"last10_metrics": last10_metrics, "trajectory_metrics": trajectory_metrics, "data_contract":',
    ),
]

for old, new in replacements:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f'expected exactly one match, found {count}: {old[:80]!r}')
    text = text.replace(old, new, 1)

path.write_text(text, encoding='utf-8')
print('patched', path)
