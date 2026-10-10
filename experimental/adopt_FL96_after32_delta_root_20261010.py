"""Original root archive/record join, only exact6 IDs/counts/CPU110/F paths rebound."""
from pathlib import Path
import ast,hashlib,json,re
from guardfed_local_storage import check_bulk_storage
ROOT=Path(__file__).resolve().parents[1]
ATTEMPT=ROOT/'tmp/celeba_flgmm_fullcoverage_delta_after32_20261010'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())
check_bulk_storage()
assert sha(ATTEMPT/'ROOT_READY_HANDOFF.json')=='fd1ca10c7ae09897599c4eb3fc8a5e3fad1f5858a3d599bfbdea6e2d5058526a'
handoff=read(ATTEMPT/'ROOT_READY_HANDOFF.json')
assert sha(ATTEMPT/'RAW_STORAGE_INDEX.json')=='b513259aedd6c0e8d43ecb403054be16fbc8bf81336f1b7d7c2f4d6273336225'
index=read(ATTEMPT/'RAW_STORAGE_INDEX.json');external=Path(index['raw_storage_root'])
assert external.resolve().drive.upper()=='F:' and len(index['files'])==72
for name,pin in index['files'].items():
    path=external/name
    assert path.resolve().is_relative_to(external.resolve()) and sha(path)==pin['sha256'] and path.stat().st_size==pin['bytes']
assert sha(ATTEMPT/'SAVED_TENSOR_STATE_CHECK.json')==handoff['full_saved_tensor_check_sha256']=='df6c67602830de52103a3ea044998ed46c9273f9e3549f612609c860999b45c0'
assert (handoff['tensor_count'],handoff['tensor_elements'])==(48,561036)
assert handoff['source_data_before_after_exact'] and handoff['CPU110_released']
assert handoff['training_calls']==handoff['CNN_forward_calls']==0
original=ROOT/'tmp/adopt_FL96_after14_delta_root_20261010.py'
assert sha(original)=='102d484268adb968ca33b127c7647ae9af64e2d3fc8ecef953b95fccc26484a9'
text=original.read_text('utf8')
ids=[f'FLGMM_Tg20_L2.0_lr0.001_IID_S-DFA_seed{s}_fullcoverage' for s in range(91005,91011)]
bindings={
 'celeba_flgmm_fullcoverage_delta_after14_20261010':ATTEMPT.name,
 "BATCH = ATTEMPT / 'batch'":'BATCH = Path('+repr(external.as_posix())+')',
 '0b07017f3ab52fdec86e7924b6f8e45a3055c25e774f0230448a5c78dc36bd6d':sha(ATTEMPT/'ROOT_READY_HANDOFF.json'),
 '5d3a886304504c898a4f70067302599042db72c0fb163dfe18c3f0bb8c1b48d6':'207f662ee41db01d75cd96a3b2ed57f84b212c0ce04ecfc34c06ae48fb99357c',
 '68a6f8d9aa7d4716094a999cfe0938b6d45ec7b4a4223f3f6a85bfcff451737e':'de54449c1fda3f93ca6a5382f075a5fd9f6676497e06e349d47f1a6cd5f75533',
 '801ec992899529dc7b38e68acfde3b101d2b90a7966def64f0bbf901cf88793b':'ae1e65bf764f3ed3e6657e95fa3798617788c8659ac30eb542035a0031663cf3',
 '2686c5c4d1121611d888462f1e288720465237ae0b80b069fc9e03a38a50513c':'254f5b844f5f302a320c5eaaa33efe790e53337c69ee31bae0f237361b5472b6',
 '8ca5e7afa10527fd01607082b0a461f0a226a4941091f182ba8fd966cd09197d':'b508fd4d0f0d36c05e4ea145ff960551d95d415b79db6c19f4d2da95935da5cf',
 "expected = [f'FLGMM_Tg20_L2.0_lr0.001_IID_F Flip_seed{s}_fullcoverage' for s in (91006, 91007)]":'expected = '+repr(ids),
 "latest['accepted_total'] == 14":"latest['accepted_total'] == 32",
 "handoff['accepted_new'] == 2":"handoff['accepted_new'] == 6",
 "handoff['accepted_new_cumulative'] == 16":"handoff['accepted_new_cumulative'] == 38",
 "handoff['archive_members'] == 27":"handoff['archive_members'] == 67",
 'accepted_before=14, accepted_new=2, accepted_total=16':'accepted_before=32, accepted_new=6, accepted_total=38',
 'archive_members_verified=27':'archive_members_verified=67',
 'next_latest = dict(accepted_total=16,':'next_latest = dict(accepted_total=38,',
 "(BATCH/'OFFSERVER_ACCEPTANCE.json').relative_to(ROOT).as_posix()":"(BATCH/'OFFSERVER_ACCEPTANCE.json').as_posix()",
 " == handoff['original_helper_seal_sha256']":'',
 'old_models_repacked=0, root_new_CNN=0,':'archive_local_path=(BATCH/"accepted_delta.tar.gz").as_posix(), raw_storage_index_sha256="b513259aedd6c0e8d43ecb403054be16fbc8bf81336f1b7d7c2f4d6273336225", helper_CPU=110, old_models_repacked=0, root_new_CNN=0,',
}
for key in bindings:assert text.count(key)==1,key
bound=re.sub('|'.join(re.escape(k) for k in sorted(bindings,key=len,reverse=True)),lambda m:bindings[m.group()],text)
ast.parse(bound)
assert not (ATTEMPT/'ROOT_ADOPTION_REVIEW.json').exists()
exec(compile(bound,str(original),'exec'),{'__file__':str(original),'__name__':'__main__'})
