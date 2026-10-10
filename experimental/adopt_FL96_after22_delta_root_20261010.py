"""Apply the original FL96 adoption checks to six new records stored only on F."""
from pathlib import Path
import ast, hashlib, json, re
from guardfed_local_storage import check_bulk_storage

ROOT = Path(__file__).resolve().parents[1]
ATTEMPT = ROOT/'tmp/celeba_flgmm_fullcoverage_delta_after22_20261010'
H = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
R = lambda p: json.loads(p.read_bytes())
check_bulk_storage()
assert H(ATTEMPT/'ROOT_READY_HANDOFF.json') == 'e988154dc85a38f9a9c6458d04cab406f22904ca0d341f1aab5b4dd38aabeba0'
handoff = R(ATTEMPT/'ROOT_READY_HANDOFF.json')
assert H(ATTEMPT/'RAW_STORAGE_INDEX.json') == 'eaf67626a41150bd337a4dc2eddb3e30b877088f1d8cd0ce46f453dff13f6029'
index = R(ATTEMPT/'RAW_STORAGE_INDEX.json'); external = Path(index['raw_storage_root'])
assert external.resolve().drive.upper() == 'F:' and len(index['files']) == 72
for name, row in index['files'].items():
    f = external/name
    assert f.resolve().is_relative_to(external.resolve()) and H(f) == row['sha256']
    assert f.stat().st_size == row['bytes']
assert H(ATTEMPT/'SAVED_TENSOR_STATE_CHECK.json') == handoff['full_saved_tensor_check_sha256'] == 'fe0c971adc1dda85e880a169a72a129be65433cc757612c52ff2dd0e0ceb1ab0'
assert (handoff['tensor_count'], handoff['tensor_elements']) == (48, 561036)
assert handoff['source_data_before_after_exact'] and handoff['CPU106_released']
assert handoff['training_calls'] == handoff['CNN_forward_calls'] == 0

original = ROOT/'tmp/adopt_FL96_after14_delta_root_20261010.py'
assert H(original) == '102d484268adb968ca33b127c7647ae9af64e2d3fc8ecef953b95fccc26484a9'
old = original.read_text('utf8')
ids = [f'FLGMM_Tg20_L2.0_lr0.001_IID_FedSA_seed{s}_fullcoverage' for s in range(91004,91010)]
bindings = {
 'celeba_flgmm_fullcoverage_delta_after14_20261010':'celeba_flgmm_fullcoverage_delta_after22_20261010',
 "BATCH = ATTEMPT / 'batch'":'BATCH = Path('+repr(external.as_posix())+')',
 '0b07017f3ab52fdec86e7924b6f8e45a3055c25e774f0230448a5c78dc36bd6d':'e988154dc85a38f9a9c6458d04cab406f22904ca0d341f1aab5b4dd38aabeba0',
 '5d3a886304504c898a4f70067302599042db72c0fb163dfe18c3f0bb8c1b48d6':'5ee1309eee278f191cef5e1a55d7fca4dc6c3336cdbf5f72cfd4c8c60f50e794',
 '68a6f8d9aa7d4716094a999cfe0938b6d45ec7b4a4223f3f6a85bfcff451737e':'9392a86efee1d27f53d4ccf661a13d2376b2f4cab8283e48647161d2930de859',
 '801ec992899529dc7b38e68acfde3b101d2b90a7966def64f0bbf901cf88793b':'31a1e0d16f14a1855acad3654d656ccb84377b7e6bd8fc3d9c10bf560658c8bc',
 '2686c5c4d1121611d888462f1e288720465237ae0b80b069fc9e03a38a50513c':'15db13ee3c35f49d9fc11b8eaf37d5ef7f360e335359bf6362a388cd496de341',
 '8ca5e7afa10527fd01607082b0a461f0a226a4941091f182ba8fd966cd09197d':'bf84b750013d143dacc16b93fb2094339c79483e8951be7b07a8f9a317915491',
 "expected = [f'FLGMM_Tg20_L2.0_lr0.001_IID_F Flip_seed{s}_fullcoverage' for s in (91006, 91007)]":'expected = '+repr(ids),
 "latest['accepted_total'] == 14":"latest['accepted_total'] == 22",
 "handoff['accepted_new'] == 2":"handoff['accepted_new'] == 6",
 "handoff['accepted_new_cumulative'] == 16":"handoff['accepted_new_cumulative'] == 28",
 "handoff['archive_members'] == 27":"handoff['archive_members'] == 67",
 'accepted_before=14, accepted_new=2, accepted_total=16':'accepted_before=22, accepted_new=6, accepted_total=28',
 'archive_members_verified=27':'archive_members_verified=67',
 'next_latest = dict(accepted_total=16,':'next_latest = dict(accepted_total=28,',
 "(BATCH/'OFFSERVER_ACCEPTANCE.json').relative_to(ROOT).as_posix()":"(BATCH/'OFFSERVER_ACCEPTANCE.json').as_posix()",
 " == handoff['original_helper_seal_sha256']":'',
 'old_models_repacked=0, root_new_CNN=0,':'archive_local_path=(BATCH/"accepted_delta.tar.gz").as_posix(), raw_storage_index_sha256="eaf67626a41150bd337a4dc2eddb3e30b877088f1d8cd0ce46f453dff13f6029", old_models_repacked=0, root_new_CNN=0,',
}
for key in bindings: assert old.count(key) == 1, key
pattern = '|'.join(re.escape(k) for k in sorted(bindings,key=len,reverse=True))
bound = re.sub(pattern,lambda m:bindings[m.group()],old)
ast.parse(bound)
assert not (ATTEMPT/'ROOT_ADOPTION_REVIEW.json').exists()
exec(compile(bound,str(original),'exec'),{'__file__':str(original),'__name__':'__main__'})
