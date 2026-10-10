"""Late binding only: original adopted64 -> selected source -> exact14 -> 2x96.

No command here dispatches a worker. Selection is unreachable before root-adopted64.
"""
from pathlib import Path
import argparse, ast, copy, difflib, hashlib, importlib.util, json, os, subprocess, sys
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
COVER = ROOT / 'tmp/celeba_gradient_fullcoverage_prepare_20261010'
SCREEN = ROOT / 'tmp/celeba_gradient_screen64_v2_20261010'
FIRST_ROOT = ROOT / 'tmp/root_adopt_first_closed_20261010/GRADIENT1_ROOT_ADOPTION.json'
FIRST_ROOT_SHA = 'ca4a38076e5080d94069cdabd50a032d84a96a339914761ec568db44a43db978'
FIRST_OFF = ROOT / 'tmp/celeba_gradient64_first_closed_adoption_20261010/OFFSERVER_ACCEPTANCE.json'
FIRST_OFF_SHA = '35089298546d72ef6b6503b63c7887912c8cd88e97a87d297d766245193e7cfa'
REMOTE_RUNS = '/workspace/celeba_gradient_screen64_v2_results_20261010/'
PINS = {'prepare.py': '7f179f38d6c3b4dddd37522d67f2e1e0065fc421e5e3fb6ddf2e4419ad3735ec',
        'screen_seal': '11e2ae63c87e0440669a047c930f5465b13babf559cca17bd8808bf693ce7ced'}
H = lambda b: hashlib.sha256(b).hexdigest()
read = lambda p: json.loads(Path(p).read_bytes())


def need(ok, message):
    if not ok: raise ValueError(message)


def pin(path, expected):
    path = Path(path)
    if not path.is_absolute(): path = ROOT / path
    need(H(path.read_bytes()) == expected, 'Changed input: ' + str(path))
    return path


def save(path, value):
    with Path(path).open('x', encoding='utf8', newline='\n') as f:
        json.dump(value, f, indent=2, allow_nan=False); f.write('\n')


def source():
    need(not sys.flags.optimize and not os.environ.get('PYTHONOPTIMIZE'), 'Unoptimized Python required')
    pin(COVER / 'prepare.py', PINS['prepare.py'])
    pin(SCREEN / 'FILES_SHA256.json', PINS['screen_seal'])
    spec = importlib.util.spec_from_file_location('original_coverage_metadata', COVER / 'prepare.py')
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def fresh_f_read_guard():
    need(sys.platform == 'win32', 'Accepted local chain reader requires Windows F storage; no path substitution')
    cmd = ['powershell', '-NoProfile', '-Command',
           "Get-Volume -DriveLetter F | Select-Object DriveLetter,FileSystemLabel,HealthStatus,SizeRemaining | ConvertTo-Json -Compress"]
    v = json.loads(subprocess.check_output(cmd, text=True))
    need(v['FileSystemLabel'] == 'Yanan 2TB' and v['HealthStatus'] == 'Healthy'
         and v['SizeRemaining'] >= 1024**3, 'F label/health/reserve guard failed')
    return v


def require64(root):
    need(root.get('accepted_total') == 64 and len(root.get('accepted_ids', [])) == 64
         and len(set(root['accepted_ids'])) == 64, 'All64 root-adopted identities required before selection')


def summary64(root_path, root_sha, out):
    """Translate adopted compact JSON; never load a checkpoint or rerun strict."""
    root_path = pin(root_path, root_sha); final = read(root_path); require64(final)
    need(not Path(out).exists(), 'Preserve existing/partial summary')
    m = source(); protocol, manifest, _, score = m.inputs()
    original = {e['id']: e for e in manifest['jobs']}
    need(set(final['accepted_ids']) == set(original), 'Foreign/missing screen ID')
    volume = fresh_f_read_guard()
    chain = []; p, sha = root_path, root_sha
    while sha != FIRST_ROOT_SHA:
        r = read(pin(p, sha)); n = r['accepted_new']
        need(r['status'] == f'ROOT_GRADIENT64_EXACT{n}_ORIGINAL_STRICT_OFFSERVER_ADOPTED'
             and r['final_test'] is False and r['new_CNN'] == r['new_training'] == 0, 'Unknown root delta schema/scope')
        h = read(pin(r['handoff_path'], r['handoff_sha256']))
        need(r['offserver_sha256'] == h['offserver_sha256'] and r['previous_root_sha256'] == h['prior_root_sha256'], 'Root/handoff chain mismatch')
        off = read(pin(h['offserver_path'], r['offserver_sha256']))
        chain.append((r, off, sha))
        p, sha = h['prior_root_path'], h['prior_root_sha256']
    first = read(pin(FIRST_ROOT, FIRST_ROOT_SHA)); off = read(pin(FIRST_OFF, FIRST_OFF_SHA))
    need(first['status'] == 'ROOT_FIRST_GRADIENT64_ORIGINAL_STRICT_ARCHIVE_AND_OFFSERVER_ADOPTED'
         and first['offserver_proof_sha256'] == FIRST_OFF_SHA and first['test'] is False, 'First1 root changed')
    chain.append((dict(accepted_before=0, accepted_new=1, accepted_total=1,
        accepted_ids=first['accepted_ids'], accepted_new_ids=first['accepted_ids']), off, FIRST_ROOT_SHA))
    rows = {}; cumulative = []; previous_root = None; previous_off = None; bindings = []
    for r, off, rsha in reversed(chain):
        ids = off['accepted_new_ids']; new = [dict(off, id=ids[0])] if r['accepted_before'] == 0 else off['records']
        off_sha = FIRST_OFF_SHA if r['accepted_before'] == 0 else r['offserver_sha256']
        need(r['accepted_before'] == len(cumulative) and r['accepted_new'] == len(ids)
             and r['accepted_total'] == len(cumulative) + len(ids)
             and r['accepted_new_ids'] == ids and not set(ids) & set(cumulative), 'Non-incremental/duplicate root identities')
        need(r['accepted_ids'] == cumulative + ids and [x['id'] for x in new] == ids, 'Ordered record/root prefix changed')
        if previous_root is not None:
            need(r['previous_root_sha256'] == previous_root and off['previous_root_sha256'] == previous_root
                 and off['previous_offserver_sha256'] == previous_off, 'Previous adopted proof not linked')
        need(off['status'] == ('OFFSERVER_ORIGINAL_STRICT_PASS' if not cumulative else 'OFFSERVER_ORIGINAL_STRICT_DELTA_PASS')
             and off['scientific_package_sha256'] == m.SEAL
             and off['original_validator_sha256'] == '2c5d7699c6e9967d32c9080fb56b4672beb821cee0b20187c68195fb37d204e9'
             and off['archive_verification']['pass'] is True and off['new_inference'] == off['new_training'] == 0, 'Original offserver identity missing')
        restored = Path(off['restored']).resolve()
        need(restored.is_relative_to(Path('F:/YananResearchStorage/GuardFed').resolve()), 'Raw metadata must remain on F')
        invpath = pin(restored / 'backup_inventory.json', off['inventory_sha256']); inv = read(invpath)
        for accepted in new:
            identity = accepted['id']; entry = original[identity]
            job = read(pin(m.OLD / 'jobs' / entry['job'], entry['job_sha256']))
            run = restored / 'runs' / identity
            values = {}
            for name in ('result.json', 'acceptance.json'):
                member = inv['members']['runs/' + identity + '/' + name]
                fp = pin(run / name, member['sha256']); need(fp.stat().st_size == member['bytes'], 'Member size changed')
                values[name] = read(fp)
            result, acceptance = values['result.json'], values['acceptance.json']
            model_sha = inv['members']['runs/' + identity + '/model.pt']['sha256']
            need(acceptance['status'] == 'PASS' and acceptance['artifact_hashes']['model.pt'] == model_sha == accepted['checkpoint_sha256']
                 and acceptance['artifact_hashes']['result.json'] == H((run / 'result.json').read_bytes()), 'Same terminal checkpoint binding changed')
            need(result['metrics'] == accepted['metrics'] and result['provenance'] == accepted['original_training_provenance']
                 and result['data_contract'] == accepted['data_contract'] and result['rounds'] == accepted['rounds'] == 70, 'Adopted record changed')
            need(result['provenance']['job_sha256'] == entry['job_sha256'] == accepted['job_sha256'], 'Original frozen job bytes required')
            contract = result['data_contract']['image_data_contract']
            need(contract['evaluation_split'] == 'valid' and contract['actual_train_rows'] == 162770
                 and contract['actual_evaluation_rows'] == 19867 and contract['train_eval_disjoint'] is True
                 and contract['root_client_disjoint'] is True, 'Original data/valid contract changed')
            for key in ('source_hashes', 'component_hashes', 'local_hashes'):
                need(result['provenance'][key] == job[key], 'Changed ' + key)
            row = dict(id=identity, method=result['method'], candidate=result['tuning_candidate'], distribution=result['distribution'],
                attack=result['attack'], seed=result['seed'], rounds=result['rounds'], alpha=result['alpha'], evaluation_split='valid',
                job_sha256=entry['job_sha256'], strict_pass=True, offserver_verified=True, metrics=result['metrics'],
                result_sha256=H((run / 'result.json').read_bytes()), model_sha256=model_sha,
                acceptance_sha256=H((run / 'acceptance.json').read_bytes()), output=REMOTE_RUNS + identity,
                source_hashes=job['source_hashes'], component_hashes=job['component_hashes'], local_hashes=job['local_hashes'],
                accepted_root_sha256=rsha, offserver_sha256=off_sha, constant_negative_retained=accepted.get('constant_negative_retained', False))
            rows[identity] = row
        cumulative += ids; previous_root, previous_off = rsha, off_sha
        bindings.append(dict(root_sha256=rsha, offserver_sha256=off_sha, inventory_sha256=off['inventory_sha256'], new_ids=ids))
    need(cumulative == final['accepted_ids'] and len(rows) == 64, 'All64 adopted chain required')
    records = [rows[e['id']] for e in manifest['jobs']]
    winners, rank = m.selected(records, protocol, manifest, score)  # Original formula/tie, only after all64.
    candidate_metric_means = {}; accuracy_champions = {}; pareto_candidates = {}
    for method in m.METHODS:
        means = {}
        for _, candidate in rank[method]:
            condition_rows = sorted((r for r in records if r['candidate'] == candidate),
                                    key=lambda r: (r['distribution'], r['attack']))
            means[candidate] = {metric: sum(r['metrics'][metric] for r in condition_rows) / 4
                                for metric in ('accuracy', 'aeod', 'aspd')}
        candidate_metric_means[method] = means
        accuracy_champions[method] = min(means, key=lambda candidate: (-means[candidate]['accuracy'], candidate))
        pareto_candidates[method] = sorted(candidate for candidate, value in means.items()
            if not any(other != candidate and rival['accuracy'] >= value['accuracy']
                       and rival['aeod'] <= value['aeod'] and rival['aspd'] <= value['aspd']
                       and (rival['accuracy'] > value['accuracy'] or rival['aeod'] < value['aeod']
                            or rival['aspd'] < value['aspd']) for other, rival in means.items()))
    save(out, dict(status='COMPLETE64_ADOPTED_METADATA_AND_SELECTION_PROPOSAL_NOT_ROOT_SELECTION_ADOPTION', records=records,
        selected_candidates=winners, all_candidate_ranks=rank, native_root_path=str(root_path.resolve()), native_root_sha256=root_sha,
        all_candidate_metric_means=candidate_metric_means, accuracy_champions=accuracy_champions, pareto_candidates=pareto_candidates,
        accepted_batch_bindings=bindings, volume_read_guard=volume, test=False, execution_authorized=False,
        limits=['n=1; no SD/significance', 'constant and negative results retained', 'root selection receipt still required',
                'output locators are original Linux runs; consumer must recheck their saved artifact hashes']))


def freeze_source(summary, summary_sha, root, root_sha, out):
    """Root-reviewed selected source construction, not source adoption or dispatch."""
    need(sys.platform == 'linux' and Path(out).resolve().is_relative_to(Path('/workspace')), 'Bind host paths on Linux /workspace only')
    m = source(); m.bind(summary, summary_sha, root, root_sha, out)
    out = Path(out); original, manifest, _, _ = m.inputs(); binding = read(out / 'BOUND_INPUTS.json')
    changes = []
    for method in m.METHODS:
        base = out / method; stage = base / 'snapshot/gradient_bridge_fullcoverage'
        protocol_path = stage / 'protocol.json'; before = protocol_path.read_bytes(); p = read(protocol_path)
        need(p['status'] == 'PREPARED_NOT_FROZEN', 'Unexpected prepared protocol')
        p['status'] = 'FROZEN'
        protocol_path.write_text(json.dumps(p, indent=2, allow_nan=False) + '\n', encoding='utf8', newline='\n')
        for name, expected in p['component_hashes'].items():
            src = pin(m.OLD / 'snapshot' / name, expected); dest = stage.parent / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            with dest.open('xb') as f: f.write(src.read_bytes())
        local = {n: H((stage / n).read_bytes()) for n in ('worker.py', 'accept_result.py', 'protocol.json')}
        mfpath = base / 'jobs/manifest.json'; mf = read(mfpath)
        for j in m.grid(p['candidates'][0], p, local):
            fp = base / 'jobs' / (j['id'] + '.json'); fp.write_text(json.dumps(j, indent=2, allow_nan=False) + '\n', encoding='utf8', newline='\n')
        for e in mf['new_jobs']: e['sha256'] = H((base / 'jobs' / e['job']).read_bytes())
        mf['status'] = 'FROZEN_SOURCE_NOT_RUNTIME_AUTHORIZED'
        mfpath.write_text(json.dumps(mf, indent=2, allow_nan=False) + '\n', encoding='utf8', newline='\n')
        for name in ('AUTHOR_DECISIONS.json', 'shared_cache_bindings.json'):
            (base / name).write_bytes((m.OLD / name).read_bytes())
        for name, src in [('run_queue.py', HERE / 'run96_queue.py'), ('coverage_contract.py', HERE / 'coverage_contract.py')]:
            (base / name).write_bytes(src.read_bytes())
        files = {f.relative_to(base).as_posix(): H(f.read_bytes()) for f in sorted(base.rglob('*')) if f.is_file()}
        save(base / 'FILES_SHA256.json', dict(files=files, execution_authorized=False))
        changes.append(dict(method=method, prepared_protocol_sha256=H(before), frozen_protocol_sha256=H(protocol_path.read_bytes()),
            manifest_sha256=H(mfpath.read_bytes()), package_seal_sha256=H((base / 'FILES_SHA256.json').read_bytes()), new_jobs=96, references=4))
    save(out / 'SOURCE_FREEZE_TRANSITION.json', dict(status='SELECTED_FROZEN_SOURCE_BUILT_ROOT_REVIEW_REQUIRED_NOT_DISPATCHED',
        bound_inputs_sha256=H((out / 'BOUND_INPUTS.json').read_bytes()), summary_sha256=summary_sha,
        selected64_root_sha256=root_sha, methods=changes, root_source_adoption_required=True, execution_authorized=False, test=False))


if __name__ == '__main__':
    a = argparse.ArgumentParser(); sub = a.add_subparsers(dest='command', required=True)
    s = sub.add_parser('summary64'); s.add_argument('--native-root', required=True); s.add_argument('--native-root-sha256', required=True); s.add_argument('--out', required=True)
    f = sub.add_parser('freeze-source'); f.add_argument('--summary', required=True); f.add_argument('--summary-sha256', required=True)
    f.add_argument('--selected64-root', required=True); f.add_argument('--selected64-root-sha256', required=True); f.add_argument('--out', required=True)
    x = a.parse_args()
    if x.command == 'summary64': summary64(x.native_root, x.native_root_sha256, x.out)
    else: freeze_source(x.summary, x.summary_sha256, x.selected64_root, x.selected64_root_sha256, x.out)
