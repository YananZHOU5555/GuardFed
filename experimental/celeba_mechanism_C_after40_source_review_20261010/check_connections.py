"""Read saved native identities and original constructor only; no model loading."""
from pathlib import Path
import ast, copy, hashlib, importlib.util, json, sys, tarfile
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent; R=H.parents[1]
B=R/'tmp/celeba_mechanism_valid_C_after40_20261010'
P=R/'tmp/celeba_mechanism_valid_C_after36_20261010'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_bytes())
def load(name,p):
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m
def save(name,obj):
    with (H/name).open('x',encoding='utf8') as f:json.dump(obj,f,indent=2);f.write('\n')

native=load('original_native147_identity_checker',B/'verify_native_snapshot.py')
# Only the output destination changes; original R/B/input/source checks stay unchanged.
native.H=H;native.main()
ni=read(B/'NATIVE_INPUTS.json'); inv=read(B/'inventory_actual147_Full100refs.json')
byid={r['id']:r for r in inv['records']}; ids=inv['selected_replay_ids']
bridge=load('review_original_bridge_metadata',B/'bridge.py')
baseline=read(R/'tmp/celeba_final_valid_replay_20261009/inputs/model_inventory.json')
manifest=read(R/'docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/manifest.json')
inspection=read(R/ni['files']['inspection']['path']);rows={r['id']:r for r in inspection['records']}
entries={r['id']:r for r in manifest['jobs']}
full={bridge.cell(r):r for r in baseline['records'] if r['method']=='GuardFed-AD2+'}
constructor_path=R/'tmp/celeba_mechanism_valid_incremental_next11_20261009/prepare.py'
nodes=[n for n in ast.walk(ast.parse(constructor_path.read_text('utf8'))) if isinstance(n,ast.Assign)
       and any(isinstance(t,ast.Name) and t.id=='record' for t in n.targets)]
assert len(nodes)==1
constructor=compile(ast.Module(body=nodes,type_ignores=[]),'unchanged_record_constructor','exec')
archive=R/ni['files']['archive']['path']
with tarfile.open(archive) as tar:
    members=json.load(tar.extractfile('backup_inventory.json'))['members']
    for model_id in ids:
        result=json.load(tar.extractfile('runs/'+model_id+'/result.json'))
        job=json.load(tar.extractfile('jobs/'+model_id+'.json'))
        row,entry=rows[model_id],entries[model_id]
        control=full[job['distribution'],job['attack'],job['config']['seed']]
        refs={k:{'archive':archive.relative_to(R).as_posix(),'archive_sha256':sha(archive),
                 'member':n,**members[n]} for k,n in [('checkpoint','runs/'+model_id+'/model.pt'),
                 ('result','runs/'+model_id+'/result.json'),('raw_job','jobs/'+model_id+'.json')]}
        ns=dict(model_id=model_id,result=result,job=job,row=row,entry=entry,control=control,refs=refs,b=bridge,copy=copy)
        exec(constructor,ns)
        assert ns['record']==byid[model_id],model_id
    assert all(rows[r['id']]==r['accepted_v4_row'] for r in inv['records'])
binding=read(B/'FULL100_ACTUAL_SOURCE_BINDINGS.json')
assert sha(R/binding['source_records_path'])==binding['source_records_sha256']
actual={r['id']:r for r in read(R/binding['source_records_path'])['records']}
prior=read(P/'FULL100_ACTUAL_SOURCE_BINDINGS.json')
assert binding['records']==prior['records'] and len(binding['records'])==100
for item in binding['records']:
    ref=item['full_reference'];r=actual[ref['id']]
    assert r['checkpoint_sha256']==ref['checkpoint_sha256']
    assert r['receipt_sha256']==item['receipt_sha256'] and r['source_binding']==item['source_binding']
assert 'torch' not in sys.modules and 'numpy' not in sys.modules
save('CONNECTIONS.json',dict(status='EXACT7_ORIGINAL_RECORD_CONSTRUCTOR_AND_FULL100_ACTUAL900_PASS',
     exact7_records_reconstructed=ids,all147_original_inspection_records_exact=True,
     Full100_binding_records_exact=True,Full100_actual900_source_records_exact=True,
     native_checker_sha256=sha(B/'verify_native_snapshot.py'),native_checker_output_only_relocated=True,
     native_review_sha256=sha(H/'ROOT_NATIVE_INCREMENT_REVIEW.json'),constructor_sha256=sha(constructor_path),
     CNN=False,torch_imported=False,numpy_imported=False))
