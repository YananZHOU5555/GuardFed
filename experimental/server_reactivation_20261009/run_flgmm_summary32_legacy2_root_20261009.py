"""Read the exact accepted legacy-two schema; original score/statistics remain byte-exact."""
from pathlib import Path
import datetime,hashlib,importlib.util,json,sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'tmp/celeba_flgmm_final6_closure_20261009'
LEGACY=ROOT/'tmp/celeba_flgmm_screen_20261009_v2_dispatch/backups/first_two_20261009/OFFSERVER_ACCEPTANCE.json'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(BASE/'summary32.py')=='52c1ad6251f1cb9d130f01ac3ecac281c7b0c7342d8cff5ce5016a341429c360'
assert sha(LEGACY)=='a53c87cbe62cdcc206dddb18cec0f62426292c05adb28a4a00de03fa64fe3750'
legacy=json.loads(LEGACY.read_bytes())
assert legacy['status']=='PARTIAL_ACCEPTED_OFFSERVER_VERIFIED' and legacy['accepted']==2
assert legacy['original_checked_result_replayed_locally'] and not legacy['final_test'] and not legacy['formal100']
assert 'different_host_observed' not in legacy
sys.path.insert(0,str(BASE))
spec=importlib.util.spec_from_file_location('frozen_summary32',BASE/'summary32.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
original=module.same_records
def same_records(server,offserver):
    if offserver==legacy:
        # The SHA-pinned first-two schema predates the host-observation field.
        assert len(server['records'])==len(offserver['records'])==2
        assert server['before_source_data']==server['after_source_data']==json.loads((BASE/'protocol.json').read_bytes())['source_hashes']
        a={r['id']:r for r in server['records']};b={r['id']:r for r in offserver['records']}
        assert len(a)==len(b)==2 and set(a)==set(b)
        for key,row in a.items():
            assert all(row[name]==b[key][name] for name in ('rounds','seed','distribution','attack','metrics','checkpoint_sha256'))
        return list(a.values())
    return original(server,offserver)
module.same_records=same_records
module.main()
