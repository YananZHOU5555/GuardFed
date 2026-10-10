"""Recover the exact already-executed source by hash; do not execute it."""
from pathlib import Path
import hashlib, itertools, json, difflib

root = Path(__file__).resolve().parents[1]
source = root / 'tmp/update_reactivation_state_20261009.py'
out = root / 'tmp/F10_entries_root_execution_20261011'
current = source.read_bytes()
h = lambda b: hashlib.sha256(b).hexdigest()
assert h(current) == '323ff72bc4c5ceae40903632d1cfde4fd08468e689e31444540048f6ad94c73a'
target = json.loads((out / 'COMMAND.json').read_bytes())['source_sha256']
assert target == '4557083549480a1d1ce74391cdeebfb10110fa18dc0e8e832ad3714053af60c7'
changes = [
    (",new_started=measured['main_terminal']+measured['main_active'],new_started_observation_utc=measured['utc']", ''),
    ("    state['server_reactivation_20261009'].update(new_formal_training=measured['main_terminal']+measured['main_active'],new_formal_training_observation_utc=measured['utc'],latest_live_scope='Historical main-only resource sample; latest_five_queue_readonly_observation supplies current queue counts.')\n", ''),
    ("    state['last_health_check_scope']='Historical main-only resource sample at its checked_utc; latest queue observations are in latest_five_queue_readonly_observation, which does not add scientific acceptance.'\n", ''),
    ("print(json.dumps({'status':phase,'measured_utc':five.get('utc',live['checked_utc']),'completed':five.get('main_terminal',live['queue_completed']),'active':five.get('main_active',len(live['active'])),'historical_resource_sample_utc':live['checked_utc'],'formal':formal}))", "print(json.dumps({'status':phase,'measured_utc':live['checked_utc'],'completed':live['queue_completed'],'active':len(live['active']),'formal':formal}))"),
]
text = current.decode('utf8')
for new, old in changes:
    assert text.count(new) == 1
matches = []
for flags in itertools.product((False, True), repeat=len(changes)):
    candidate = text
    for use, (new, old) in zip(flags, changes):
        if use:
            candidate = candidate.replace(new, old, 1)
    b = candidate.encode('utf8')
    if h(b) == target:
        matches.append((flags, b))
assert len(matches) == 1, ('no exact historical source recovered', len(matches))
flags, executed = matches[0]
snapshot = out / 'EXECUTED_SOURCE_45570835.py'
assert not snapshot.exists()
snapshot.write_bytes(executed)
diff = ''.join(difflib.unified_diff(executed.decode('utf8').splitlines(keepends=True), text.splitlines(keepends=True), fromfile='executed_source_45570835', tofile='future_source_323ff72b'))
(out / 'POST_EXECUTION_SOURCE_METADATA_DIFF.patch').write_bytes(diff.encode('utf8'))
proof = dict(status='EXACT_EXECUTED_SOURCE_RECOVERED_BY_COMMAND_HASH_NOT_REEXECUTED', executed_source_sha256=h(executed), future_source_sha256=h(current), reversed_spans=list(flags), bounded_candidates=2**len(changes), command_source_sha256=target, command_exit=json.loads((out/'EXIT.json').read_bytes())['returncode'], source_snapshot=snapshot.relative_to(root).as_posix(), diff_sha256=h(diff.encode('utf8')), current_source_unchanged=source.read_bytes()==current, generator_reexecuted=False, scientific_result_changed=False)
(out / 'EXECUTED_SOURCE_RECOVERY.json').write_text(json.dumps(proof, indent=2)+'\n', encoding='utf8')
print(json.dumps(proof))
