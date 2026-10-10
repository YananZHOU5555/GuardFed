"""Local source/metadata fixtures only. Synthetic future objects remain in memory."""
from pathlib import Path
import ast,copy,hashlib,importlib.util,json
H=Path(__file__).resolve().parent;R=H.parents[1];P=R/'tmp/celeba_remaining_F_FedSA10_20261011'
read=lambda p:json.loads(Path(p).read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def loop(path,target):
    s=path.read_text('utf8');t=ast.parse(s)
    candidates=[n for n in ast.walk(t) if isinstance(n,ast.For) and ast.unparse(n.target)==target]
    assert len(candidates)==1
    return ast.get_source_segment(s,candidates[0])
def main():
    pins=read(H/'PARENT_SOURCE_PINS.json')
    for n,v in pins['files'].items():assert sha(P/n)==v['sha256'] and (P/n).stat().st_size==v['bytes']
    names=list(pins['files'])+['prepare_source.py','check_source.py']
    for n in names:compile((H/n).read_bytes(),str(H/n),'exec')
    exact=['execute_once.py','remote_cpu111_export.py','cpu_release_readonly.py']
    for n in exact:assert (P/n).read_bytes()==(H/n).read_bytes()
    a=loop(P/'join_saved.py','rec');b=loop(H/'join_saved.py','rec');assert a==b
    assert loop(P/'adopt_by_root.py','rec')==loop(H/'adopt_by_root.py','rec')
    assert loop(P/'adopt_by_root.py','na').replace("row['attack']=='FedSA'","row['attack']=='S-DFA'")==loop(H/'adopt_by_root.py','na')
    marker='assert os.sched_getaffinity(0)=={111}'
    assert (P/'remote_preflight.py').read_text('utf8').split(marker)[1]==(H/'remote_preflight.py').read_text('utf8').split(marker)[1]
    p=read(H/'PREPARED.json')
    assert p['native_binding'] is p['authorized_snapshot_path'] is p['authorized_snapshot_sha256'] is p['native_target_total'] is None
    assert p['required_native_root_path'] is p['required_native_root_sha256'] is None and not p['execution_performed']
    for f in ('EXECUTION_INPUTS.json','PREFLIGHT_COMMAND.json','RAW_STORAGE_INDEX.json','MECHANISM_INDEX.json','ROOT_ACTUAL_PINS.json'):
        assert not (H/f).exists()
    # No files representing a future proof are created, and no physical F path is read.
    objects={p[k+'_path']:read(p[k+'_path']) for k in ('prior_root','prior_index','native333_root','native333_inspection','native333_ledger')}
    hashes={p[k+'_path']:p[k+'_sha256'] for k in ('prior_root','prior_index','native333_root','native333_inspection','native333_ledger')}
    for path,w in hashes.items():assert sha(path)==w
    old=objects[p['native333_inspection_path']];oldledger=objects[p['native333_ledger_path']]
    newrows=[dict(id=i,variant='minus_F',distribution='IID',attack='S-DFA',seed=int(i[-5:]),prediction_support={'prediction_count':19867}) for i in p['required_new_native_candidate_ids']]
    objects.update({'MEMORY_ONLY_FUTURE_ROOT':dict(root_adopted=True,total_new_strict_and_offserver=340,test=False,inspection_sha256='SYNTHETIC_INSPECTION',ledger_sha256='SYNTHETIC_LEDGER',previous_ledger_sha256=p['native333_ledger_sha256'],new_ids=p['required_new_native_candidate_ids']),
        'MEMORY_ONLY_FUTURE_INSPECTION':dict(records=copy.deepcopy(old['records'])+newrows,accepted_new_ids=old['accepted_new_ids']+p['required_new_native_candidate_ids']),
        'MEMORY_ONLY_FUTURE_LEDGER':dict(entries=copy.deepcopy(oldledger['entries'])+[{'synthetic_fixture':True}]),
        'MEMORY_ONLY_FUTURE_SNAPSHOT':dict(remaining620=dict(remote_closed_ids=list(p['candidate_ids']),failures=[]),source_data_binding=dict(changed_paths=[]))})
    hashes.update(dict(MEMORY_ONLY_FUTURE_ROOT='SYNTHETIC_ROOT',MEMORY_ONLY_FUTURE_INSPECTION='SYNTHETIC_INSPECTION',MEMORY_ONLY_FUTURE_LEDGER='SYNTHETIC_LEDGER',MEMORY_ONLY_FUTURE_SNAPSHOT='SYNTHETIC_SNAPSHOT'))
    e=dict(native_root_verified=True,exact_candidate_ids=p['candidate_ids'],native_root_path='MEMORY_ONLY_FUTURE_ROOT',native_root_sha256='SYNTHETIC_ROOT',
      native_inspection_path='MEMORY_ONLY_FUTURE_INSPECTION',native_inspection_sha256='SYNTHETIC_INSPECTION',native_ledger_path='MEMORY_ONLY_FUTURE_LEDGER',native_ledger_sha256='SYNTHETIC_LEDGER',root_native_total=340,transport_target_only=10,proposed_replay_total=340,
      native_archive_roots=p['known_native_archive_roots']+[dict(path='MEMORY_ONLY_FUTURE_ROOT',sha256='SYNTHETIC_ROOT')],authorized_snapshot_path='MEMORY_ONLY_FUTURE_SNAPSHOT',authorized_snapshot_sha256='SYNTHETIC_SNAPSHOT')
    spec=importlib.util.spec_from_file_location('metadata_only_sd_fixtures',H/'verify_native_inputs.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    m.sha=lambda path:hashes[str(path)];m.read=lambda path:objects[str(path)]
    assert m.validate(e,p)==e
    failures=[]
    def reject(name,mutate):
        base=copy.deepcopy(objects);ee=copy.deepcopy(e);pp=copy.deepcopy(p)
        mutate(ee,pp,objects)
        try:m.validate(ee,pp)
        except (AssertionError,KeyError,TypeError):failures.append(name)
        else:raise AssertionError('Fixture improperly accepted: '+name)
        finally:objects.clear();objects.update(base)
    reject('no_actual_native_proof',lambda e,p,o:e.update(native_root_verified=False))
    reject('native333_only',lambda e,p,o:e.update(root_native_total=333))
    reject('wrong_native_source_sha',lambda e,p,o:e.update(native_root_sha256='WRONG'))
    reject('missing_seventh_new_native',lambda e,p,o:o['MEMORY_ONLY_FUTURE_ROOT']['new_ids'].pop())
    reject('changed_old433_record',lambda e,p,o:o['MEMORY_ONLY_FUTURE_INSPECTION']['records'][0].update(id='altered'))
    reject('changed_old47_ledger_prefix',lambda e,p,o:o['MEMORY_ONLY_FUTURE_LEDGER']['entries'][0].update(changed=True))
    reject('missing_prior_native_archive',lambda e,p,o:e['native_archive_roots'].pop(0))
    reject('wrong_attack',lambda e,p,o:o['MEMORY_ONLY_FUTURE_INSPECTION']['records'][-1].update(attack='FedSA'))
    reject('missing_remote_closure',lambda e,p,o:o['MEMORY_ONLY_FUTURE_SNAPSHOT']['remaining620']['remote_closed_ids'].pop())
    reject('remote_failure_evidence',lambda e,p,o:o['MEMORY_ONLY_FUTURE_SNAPSHOT']['remaining620']['failures'].append('failure'))
    reject('source_changed',lambda e,p,o:o['MEMORY_ONLY_FUTURE_SNAPSHOT']['source_data_binding']['changed_paths'].append('source.py'))
    reject('wrong_snapshot_sha',lambda e,p,o:e.update(authorized_snapshot_sha256='WRONG'))
    return dict(status='SOURCE_ONLY_COMPILE_SCIENCE_LOOP_IDENTITY_AND_SYNTHETIC_METADATA_REJECTIONS_PASS',
      compiled_sources=len(names),exact_original_helpers=exact,join_per_record_loop_raw_exact=True,
      adopter_saved_per_record_loop_raw_exact=True,adopter_native_loop_exact_after_attack_literal=True,
      CPU111_resource_source_failure_closure_guards_raw_exact=True,
      original_saved_verifier_and_transport_unchanged=True,synthetic_positive_fixtures=1,synthetic_rejections=failures,
      synthetic_objects_written=False,actual_native_binding=None,actual_snapshot=None,execution_performed=False,
      SSH=False,F_bulk_read=False,new_CNN=0,new_fit=0,new_training=0,test=False,
      parent_root_sha256=p['prior_root_sha256'],parent_index_sha256=p['prior_index_sha256'],
      source_files={n:dict(sha256=sha(H/n),bytes=(H/n).stat().st_size) for n in names})
if __name__=='__main__':print(json.dumps(main(),ensure_ascii=False,indent=2))
