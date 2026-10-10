"""Correct five current writer labels; preserve all historical sections."""
from pathlib import Path
import datetime, hashlib, json, os

R = Path(__file__).resolve().parents[1]
T = R / 'docs/server_deployment_20260923/training_20260923'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
sp = T / 'TRAINING_STATE.json'
assert sha(sp) == 'e95c4c7728e424cad72a22901f597f300b8d534abc7a7704fecb3db5b01b6534'
items = [(T/'RUNNING.md', b'# HISTORICAL:'),
 (T/'REBUTTAL_COMPLETION_20261009.md', b'## Historical accepted increment'),
 (R/'docs/返修实验总览.md', '以下为 2026-10-04'.encode()),
 (T/'server_reactivation_20261009/MONITOR_HANDOFF.md', b'# Historical handoff snapshots'),
 (T/'celeba_mechanism_v1/EXECUTION.md', b'# HISTORICAL PREPARATION SNAPSHOT')]
old = b'tmp/update_increment59_entries_20261011.py'
new = b'tmp/update_increment60_entries_20261011.py'
prepared = []
for p, marker in items:
    content = p.read_bytes(); before, history = content.split(marker, 1)
    assert before.count(old) == 1
    prepared.append((p, before.replace(old, new)+marker+history,
                     hashlib.sha256(marker+history).hexdigest(), sha(p)))
proof = dict(status='ROOT_CURRENT_ENTRY_WRITER_LABEL_ONLY_CORRECTED',
    utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    source_sha256=sha(__file__), entries={}, new_scientific_acceptances=0)
for p, content, history_sha, before_sha in prepared:
    p.write_bytes(content)
    proof['entries'][p.relative_to(R).as_posix()] = dict(
        before_sha256=before_sha, sha256=sha(p), history_sha256=history_sha,
        history_exact=True, writer_label_only=True)
s = json.loads(sp.read_bytes())
s['current_entry_correction'] = dict(path=Path(__file__).relative_to(R).as_posix(),
    sha256=sha(__file__), utc=proof['utc'], scope='Five current writer labels only')
target = sp.with_suffix('.label-correction.tmp'); assert not target.exists()
target.write_text(json.dumps(s, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
os.replace(target, sp)
proof['STATE_sha256'] = sha(sp)
out = R/'tmp/publication_increment60_prepared_20261011/WRITER_LABEL_CORRECTION_ACTUAL.json'
with out.open('x', encoding='utf8') as f:
    json.dump(proof, f, indent=2); f.write('\n')
print(json.dumps(dict(STATE_sha256=sha(sp), proof_sha256=sha(out))))
