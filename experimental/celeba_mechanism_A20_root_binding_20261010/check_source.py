"""Source and compact input identities only: never execute adopter, F guard or arrays."""
from pathlib import Path
import ast,collections,hashlib,json,sys
sys.dont_write_bytecode=True
from bind_A20 import ROOT,HERE,ORIGINAL,ORIGINAL_SHA,PRIOR_ROOT_SHA,PRIOR_INDEX_SHA,EXPECTED,parser,render,sha
read=lambda p:json.loads(Path(p).read_bytes())
def save(name,value):
    with (HERE/name).open('x',encoding='utf8',newline='\n') as f:json.dump(value,f,indent=2);f.write('\n')

delivery=ROOT/'tmp/celeba_remaining620_A20_transport_20261010'
seal=delivery/'FILES_SHA256.json';assert sha(seal)=='66b9e054a005786ca3d8d0e32547bbd83d7a8c66520b31b611a2bde759d6f9cc'
for name,pin in read(seal)['files'].items():
    p=delivery/name;assert sha(p)==pin['sha256'] and p.stat().st_size==pin['bytes']
storage=read(delivery/'RAW_STORAGE_INDEX.json');prepared=read(delivery/'PREPARED.json')
assert prepared['selected_ids']==storage['accepted_new_ids']==EXPECTED
assert prepared['science_transport_unchanged'] is True and 'actual_export' not in prepared and prepared['CPU']==110
assert storage['accepted_offserver']==storage['root_adopted']==0
assert (storage['metrics'],storage['counts'],storage['rules'],storage['archive_members'])==(72,192,24,74)
args=parser().parse_args(['--delivery-seal-sha256',sha(seal),'--storage-index-sha256',sha(delivery/'RAW_STORAGE_INDEX.json'),
    '--archive-sha256',storage['archive_sha256'],'--receipt-sha256',storage['receipt_sha256'],
    '--offserver-sha256',storage['offserver_verification_sha256'],'--archive-members','74'])
source,diff=render(args);original=ORIGINAL.read_text('utf8')
start="extract = Path(storage['offserver_verification']).parent / 'verified_extract'"
end='assert member_checks == '
assert original[original.index(start):original.index(end)]==source[source.index(start):source.index(end)]
assert "archive_check = v4.verify_archive(Path(storage['archive']), read(storage['receipt']))" in source
for check in ["assert rec['native_max_abs_difference'] == 0",
    "assert all(abs(x) <= 1e-12 for values in rec['differences'].values() for x in values.values())",
    "assert (rec['independent_metric_checks'], rec['independent_confusion_count_checks'], rec['prediction_rule_checks']) == (9, 24, 3)"]:
    assert check in source and check in original
tree=ast.parse(source)
for n in ast.walk(tree):
    if isinstance(n,ast.Dict):
        keys=[k.value for k in n.keys if isinstance(k,ast.Constant)];assert len(keys)==len(set(keys))
    if isinstance(n,ast.Call):
        keys=[k.arg for k in n.keywords if k.arg is not None];assert len(keys)==len(set(keys))
prior_root=ROOT/'tmp/celeba_mechanism_remaining620_A12_root_adoption_20261010/ROOT_ADOPTION.json'
prior_index=prior_root.parent/'MECHANISM212_INDEX.json'
assert sha(prior_root)==PRIOR_ROOT_SHA and sha(prior_index)==PRIOR_INDEX_SHA
prior=read(prior_index);assert len(prior['all_ids'])==len(set(prior['all_ids']))==212
assert collections.Counter(x.split('_',2)[1] for x in prior['all_ids'])=={'U':100,'C':100,'A':12}
assert not set(prior['all_ids']) & set(EXPECTED)
allids=prior['all_ids']+EXPECTED;assert len(allids)==len(set(allids))==220
assert collections.Counter(x.split('_',2)[1] for x in allids)=={'U':100,'C':100,'A':20}
assert not (ROOT/'tmp/celeba_mechanism_remaining620_A20_root_adoption_20261010').exists()
refusals=[]
for field in ('delivery_seal_sha256','storage_index_sha256','archive_sha256','receipt_sha256','offserver_sha256'):
    bad=type(args)(**vars(args));setattr(bad,field,'missing-or-malformed')
    try:render(bad)
    except AssertionError:refusals.append(field)
    else:raise AssertionError(field)
bad=type(args)(**vars(args));bad.archive_members=110
try:render(bad)
except AssertionError:refusals.append('old_A12_member_count')
else:raise AssertionError('old member count passed')
assert not ({'torch','numpy'} & sys.modules.keys())
with (HERE/'METADATA_DIFF.patch').open('x',encoding='utf8',newline='\n') as f:f.write(diff)
pins={str(p.relative_to(ROOT).as_posix()):sha(p) for p in [ORIGINAL,prior_root,prior_index,seal,delivery/'RAW_STORAGE_INDEX.json',delivery/'PREPARED.json',delivery/'HANDOFF.json']}
save('INPUTS.json',dict(status='ACTUAL_SOURCE_METADATA_PINS_NOT_ADOPTION',pins=pins,CLI_arguments=vars(args),
    expected_new_ids=EXPECTED,excluded_prior_count=212,prospective_cumulative_count=220,
    new_native_archives=['root_delta_20261010T070642Z','root_delta_20261010T072729Z'],
    original_scientific_transport_source_sha256='a3461e20592cd2bda3d53c7215fed377bbaa693360ba1abe02aa87fe8aa6fc03',
    root_adoption_executed=False))
save('SOURCE_CHECK.json',dict(status='SOURCE_ONLY_A20_BINDING_READY_NOT_EXECUTED',original_adopter_sha256=ORIGINAL_SHA,
    original_per_record_saved_metric_threshold_checkpoint_native_member_block_exact=True,
    original_archive_verifier_call_exact=True,native_tolerance_unchanged=1e-12,
    compact_delivery_seal_verified=True,expected_new_ids=EXPECTED,prior212_ids_exact=True,
    prospective220_unique=True,composition={'U':100,'C':100,'A':20},source_metadata_refusals=refusals,
    unique_AST_metadata_keys=True,AST_compile_pass=True,new_adoption_directory_exists=False,
    source_ready_not_root_adopted=True,source_body_executed=False,F_volume_queried=False,
    arrays_read=False,models_read=False,SSH=False,CNN=False,new_fitting=False,STATE_Git_written=False))
print(json.dumps(dict(status='SOURCE_ONLY_A20_BINDING_READY_NOT_EXECUTED',scope=8,prior=212,cumulative_if_adopted=220,
    source_refusals=len(refusals),original_scientific_check_block_exact=True)))
