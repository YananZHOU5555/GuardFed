"""Reuse the original FL96 root checks for the exact four-record F-only delta."""
from pathlib import Path
import ast, hashlib, json, re
from guardfed_local_storage import check_bulk_storage
ROOT=Path(__file__).resolve().parents[1]
ATTEMPT=ROOT/'tmp/celeba_flgmm_fullcoverage_delta_after28_20261010'
H=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
R=lambda p:json.loads(p.read_bytes())
check_bulk_storage()
assert H(ATTEMPT/'ROOT_READY_HANDOFF.json')=='e811aeca9b36c85cfcbebf2fad4fcc677ce835b8a834cb13d6c3f37338f28b3e'
handoff=R(ATTEMPT/'ROOT_READY_HANDOFF.json')
assert H(ATTEMPT/'RAW_STORAGE_INDEX.json')=='c2f17efc9a0bc2aed2a3836a2b5f77fe90c15d1034af1b800454ddf4b590df48'
index=R(ATTEMPT/'RAW_STORAGE_INDEX.json');external=Path(index['raw_storage_root'])
assert external.resolve().drive.upper()=='F:' and len(index['files'])==52
for name,row in index['files'].items():
    f=external/name
    assert f.resolve().is_relative_to(external.resolve()) and H(f)==row['sha256'] and f.stat().st_size==row['bytes']
assert H(ATTEMPT/'SAVED_TENSOR_STATE_CHECK.json')==handoff['full_saved_tensor_check_sha256']=='8ff0ef457b5d50c96a54bf2203980f9912e2ff2b37e607a776011f8f0a14ac0b'
assert (handoff['tensor_count'],handoff['tensor_elements'])==(32,374024)
assert handoff['source_data_before_after_exact'] and handoff['CPU106_released']
assert handoff['training_calls']==handoff['CNN_forward_calls']==0
original=ROOT/'tmp/adopt_FL96_after14_delta_root_20261010.py'
assert H(original)=='102d484268adb968ca33b127c7647ae9af64e2d3fc8ecef953b95fccc26484a9'
old=original.read_text('utf8')
ids=['FLGMM_Tg20_L2.0_lr0.001_IID_FedSA_seed91010_fullcoverage']+[
    f'FLGMM_Tg20_L2.0_lr0.001_IID_S-DFA_seed{s}_fullcoverage' for s in (91002,91003,91004)]
bindings={
 'celeba_flgmm_fullcoverage_delta_after14_20261010':'celeba_flgmm_fullcoverage_delta_after28_20261010',
 "BATCH = ATTEMPT / 'batch'":'BATCH = Path('+repr(external.as_posix())+')',
 '0b07017f3ab52fdec86e7924b6f8e45a3055c25e774f0230448a5c78dc36bd6d':'e811aeca9b36c85cfcbebf2fad4fcc677ce835b8a834cb13d6c3f37338f28b3e',
 '5d3a886304504c898a4f70067302599042db72c0fb163dfe18c3f0bb8c1b48d6':'75f1f198a3f672eb732478eba912d810d0bbd6cef059eb647e35fdbeee5c776c',
 '68a6f8d9aa7d4716094a999cfe0938b6d45ec7b4a4223f3f6a85bfcff451737e':'97e2b2d7096be6651b27c5c9a60077a29381c01317d12e37f937f554a3d3f1e5',
 '801ec992899529dc7b38e68acfde3b101d2b90a7966def64f0bbf901cf88793b':'e51007c549f8cc95970ce06cb61b4a477c11eff5d00e05a9aeac9ffb30dd449c',
 '2686c5c4d1121611d888462f1e288720465237ae0b80b069fc9e03a38a50513c':'bf84b750013d143dacc16b93fb2094339c79483e8951be7b07a8f9a317915491',
 '8ca5e7afa10527fd01607082b0a461f0a226a4941091f182ba8fd966cd09197d':'254f5b844f5f302a320c5eaaa33efe790e53337c69ee31bae0f237361b5472b6',
 "expected = [f'FLGMM_Tg20_L2.0_lr0.001_IID_F Flip_seed{s}_fullcoverage' for s in (91006, 91007)]":'expected = '+repr(ids),
 "latest['accepted_total'] == 14":"latest['accepted_total'] == 28",
 "handoff['accepted_new'] == 2":"handoff['accepted_new'] == 4",
 "handoff['accepted_new_cumulative'] == 16":"handoff['accepted_new_cumulative'] == 32",
 "handoff['archive_members'] == 27":"handoff['archive_members'] == 47",
 'accepted_before=14, accepted_new=2, accepted_total=16':'accepted_before=28, accepted_new=4, accepted_total=32',
 'archive_members_verified=27':'archive_members_verified=47',
 'next_latest = dict(accepted_total=16,':'next_latest = dict(accepted_total=32,',
 "(BATCH/'OFFSERVER_ACCEPTANCE.json').relative_to(ROOT).as_posix()":"(BATCH/'OFFSERVER_ACCEPTANCE.json').as_posix()",
 " == handoff['original_helper_seal_sha256']":'',
 'old_models_repacked=0, root_new_CNN=0,':'archive_local_path=(BATCH/"accepted_delta.tar.gz").as_posix(), raw_storage_index_sha256="c2f17efc9a0bc2aed2a3836a2b5f77fe90c15d1034af1b800454ddf4b590df48", old_models_repacked=0, root_new_CNN=0,',
}
for key in bindings:assert old.count(key)==1,key
bound=re.sub('|'.join(re.escape(k) for k in sorted(bindings,key=len,reverse=True)),lambda m:bindings[m.group()],old)
ast.parse(bound)
assert not (ATTEMPT/'ROOT_ADOPTION_REVIEW.json').exists()
exec(compile(bound,str(original),'exec'),{'__file__':str(original),'__name__':'__main__'})
