"""Pinned C_after60 operation bodies with exact after70 metadata rebindings."""
from pathlib import Path
import ast,hashlib,json,re,sys

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OLD=ROOT/'tmp/celeba_mechanism_C_after60_root_operations_20261010'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_bytes())

def bindings():
    if sys.flags.optimize:raise RuntimeError('Optimized Python is forbidden for root operation guards')
    m=read(HERE/'BINDINGS.json')
    for rel,pin in m['pins'].items():
        if sha(ROOT/rel)!=pin:raise ValueError('Pinned operation/source identity changed: '+rel)
    b=ROOT/m['prepared'];inventory=read(b/'inventory_actual180_Full100refs.json')
    prior=read(ROOT/m['prior_inventory']);scope=read(b/'SCOPE.json')
    ni=read(b/'NATIVE_INPUTS.json');receipt=read(b/'PACKAGE_RECEIPT.json')
    adoption=read(ROOT/m['prior_adoption'])
    ids=[f'minus_C_non-IID_FedSA_seed{s}' for s in range(91001,91011)]
    assert scope['selected_ids']==inventory['selected_replay_ids']==receipt['selected_ids']==ni['selected_ids']==ids
    closed=[r['id'] for r in prior['records']]
    assert len(closed)==len(set(closed))==170 and scope['excluded_prior_ids']==inventory['excluded_prior_replay_ids']==closed
    assert len(inventory['records'])==180 and {r['id'] for r in inventory['records']}==set(closed)|set(ids)
    assert [r for r in inventory['records'] if r['id'] in set(closed)]==prior['records']
    assert inventory['full_references']==prior['full_references'] and len(inventory['full_references'])==100
    assert adoption['cumulative_three_view_models']==170 and adoption['original160_unchanged'] and adoption['all_native_differences_zero']
    assert inventory['prior_replay_boundary']['root_adoption_receipt_sha256']==m['pins'][m['prior_adoption']]
    chain=inventory['backup_chain'][-1]
    assert chain['archive_sha256']==ni['files']['archive']['sha256'] and chain['accepted_new_ids']==ids
    assert all(inventory['increment_archive_sources'][i]==ni['files']['archive']['path'] for i in ids)
    assert receipt['science_seal_sha256']==m['replacements']['ed8ecc84781b799e205c9e139dce8e753cb5545139b7e48ef23d5f8d07a50a77']
    assert receipt['execution_seal_sha256']==m['replacements']['12c1b612d9c9697f6a537344ebace5aaa7c1bf5d7bbe760a3866168adca89896']
    return m

def patched_source(operation):
    if operation not in ('deploy','observe','backup','adopt'):raise ValueError('Unknown fixed operation')
    m=bindings();p=OLD/(operation+'.py')
    assert sha(p)==m['original_operations'][operation]
    source=p.read_text(encoding='utf-8-sig')
    changes=m['replacements']
    source=re.sub('|'.join(re.escape(k) for k in sorted(changes,key=len,reverse=True)),lambda match:changes[match[0]],source)
    # Only standalone count literals change; hashes, UTF8, CPU112..119 and eight threads cannot match.
    source=re.sub(r'\b(160|170)\b',lambda match:{'160':'170','170':'180'}[match[0]],source)
    ast.parse(source)
    return source

def run(operation):
    source=patched_source(operation)
    # Preserve original main/error lifecycle; observe's original body intentionally runs at top level.
    namespace={'__name__':'__main__','__file__':str(HERE/(operation+'.py')),'__package__':None}
    exec(compile(source,str(HERE/(operation+'.py')),'exec'),namespace)

if __name__=='__main__':raise SystemExit('Use deploy.py, observe.py, backup.py or adopt.py explicitly')
