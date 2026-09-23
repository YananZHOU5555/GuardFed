#!/usr/bin/env python3
import json
from collections import Counter, defaultdict
from pathlib import Path

p = Path('/home/yannan/workspace/GuardFed/results/attack_strength/raw_results.jsonl')
rows = [json.loads(line) for line in p.read_text(encoding='utf-8').splitlines() if line.strip()]
print('total', len(rows))
print('study', Counter(r.get('study') for r in rows))
main = [r for r in rows if r.get('study') == 'main']
ratio = [r for r in rows if r.get('study') == 'ratio']
print('main by method', Counter(r.get('method') for r in main))
print('ratio by dataset/attack', Counter((r.get('dataset'), r.get('attack')) for r in ratio))
print('ratio by seed', Counter(r.get('seed') for r in ratio))
print('ratio by ratio', Counter(r.get('malicious_ratio') for r in ratio))
keys = Counter((r.get('study'), r.get('dataset'), r.get('distribution'), r.get('method'), r.get('attack'), r.get('seed'), r.get('malicious_ratio')) for r in rows)
print('duplicate keys', sum(v-1 for v in keys.values() if v > 1), 'unique keys', len(keys))
print('ratio expected keys', 400, 'ratio unique', len({(r.get('dataset'), r.get('distribution'), r.get('method'), r.get('attack'), r.get('seed'), r.get('malicious_ratio')) for r in ratio}))
