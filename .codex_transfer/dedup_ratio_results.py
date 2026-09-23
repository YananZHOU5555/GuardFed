#!/usr/bin/env python3
"""Merge the main JSONL and disjoint ratio shards by run_id."""
from __future__ import annotations

import json
from pathlib import Path

results = Path('/tmp/GuardFed/results/attack_strength')
paths = [results / 'raw_results.jsonl', results / 'raw_results_ratio_shard0.jsonl', results / 'raw_results_ratio_shard1.jsonl']
seen = set()
records = []
input_count = 0
duplicates = 0
for path in paths:
    if not path.exists():
        continue
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        input_count += 1
        row = json.loads(line)
        key = row.get('run_id')
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        records.append(row)
out = results / 'raw_results_ratio_dedup.jsonl'
with out.open('w', encoding='utf-8') as handle:
    for row in records:
        handle.write(json.dumps(row, ensure_ascii=False) + '\n')
(results / 'raw_results.jsonl').replace(results / 'raw_results_before_ratio_dedup.jsonl')
out.replace(results / 'raw_results.jsonl')
print(json.dumps({'input_records': input_count, 'unique_records': len(records), 'duplicates_removed': duplicates}, indent=2))
