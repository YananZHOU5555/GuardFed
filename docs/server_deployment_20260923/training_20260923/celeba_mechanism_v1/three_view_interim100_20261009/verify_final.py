"""Finite offline source/identity/display verification of this adopted snapshot."""
import copy
import json
import sys
sys.dont_write_bytecode=True
import build as b

def main():
    prepared=b.read(b.H/'FILES_SHA256.json')
    b.need(b.sha(b.H/'FILES_SHA256.json')=='8d085d2a1e39f0c37dcaaea962a593614c055f05591d05691de7cbea0795fe40','Prepared seal changed')
    for name,pin in prepared['files'].items():
        p=b.H/name;b.need(b.sha(p)==pin['sha256'] and p.stat().st_size==pin['bytes'],'Prepared member changed '+name)
    source=b.read(b.H/'FINAL_SOURCE_BINDINGS.json')
    for name,pin in source['actual_new8_files'].items():
        p=b.R/name;b.need(b.sha(p)==pin['sha256'] and p.stat().st_size==pin['bytes'],'Actual evidence drift '+name)
    root=b.R/source['new_root_adoption'];proof=b.read(root);b.adoption_gate(proof,root,source['new_root_adoption_sha256'])
    refusals=[]
    for field,value in [('accepted_new',7),('cumulative_three_view_models',99),('original92_unchanged',False),('prior92_root_adoption_sha256','0'*64),('test_inference',True)]:
        bad=copy.deepcopy(proof);bad[field]=value
        try:b.adoption_gate(bad,root,source['new_root_adoption_sha256'])
        except ValueError:refusals.append(field)
        else:raise AssertionError('Accepted drift '+field)
    s=b.H/'snapshot100';records=b.read(s/'records.json')['records'];tables=b.read(s/'tables.json');v=b.read(s/'verification.json')
    old=b.read(b.PRIOR/'snapshot92/records.json')['records']
    assert records[:184]==old and len(records)==len({r['id'] for r in records})==200
    i100=b.read(b.P100/'inventory_actual100_Full100refs.json');inv=b.scope(b.read(b.P92/'inventory_actual92_Full100refs.json'),i100)
    full900={r['id']:r for r in b.read(b.FULL/'records_three_views_900.json')['records']}
    refs={r['id']:r for r in i100['full_references']};bycell={}
    for r in records:
        key=(r['variant'],r['distribution'],r['attack'],r['seed']);assert key not in bycell;bycell[key]=r
        if r['variant']=='Full':assert r==b.full_record(full900[r['id']],refs[r['id']])
        else:
            expected=inv[r['id']];assert r['variant']=='minus_U' and r['checkpoint_sha256']==expected['checkpoint']['sha256']
            assert r['config_sha256']==expected['config_canonical_sha256'] and r['data_contract']==expected['data_contract']
            assert max(abs(r['views']['native'][k]-expected['prior_validation_metrics'][k]) for k in ('accuracy','aeod','aspd'))<=1e-12
    assert set(r['id'] for r in records if r['variant']=='minus_U')==set(inv)
    assert set(r['id'] for r in records if r['variant']=='Full')==set(refs)
    for r in records:
        if r['variant']=='minus_U':assert r['data_contract']==bycell[('Full',r['distribution'],r['attack'],r['seed'])]['data_contract']
    arithmetic=b.module('existing_independent_fsum_check',b.H/'verify_numeric.py').verify(records,tables['panels'])
    for k,value in arithmetic.items():assert v[k]==value
    assert v['display_cells_verified']==810 and v['old_nine_scene_rows_exact']==243 and v['old_nine_display_cells_exact']==729
    lines=[s for s in (s/'TABLES.md').read_text(encoding='utf8').splitlines() if s.startswith('| ') and ' ± ' in s]
    cells=0
    for line,row in zip(lines,[r for p in tables['panels'] for r in p['rows']]):
        columns=line.strip('| ').split(' | ')
        for index,key in enumerate(('accuracy_pct','aeod','aspd'),4):
            digits=3 if key=='accuracy_pct' else 5
            assert columns[index]==f"{row[key]['mean']:.{digits}f} ± {row[key]['sample_sd_ddof1']:.{digits}f}";cells+=1
    assert len(lines)==270 and cells==810
    assert tables['native_shared_identical_records']==sum(r['views']['native']==r['views']['shared_calibration'] for r in records)==200
    assert not tables['final_test'] and not tables['primary_endpoint_selected'] and tables['new_inference']==tables['new_training']==0
    result=dict(status='FINAL_OFFLINE_SOURCE_IDENTITY_COUNTS_STATISTICS_DISPLAY_PASS_PENDING_ROOT_REVIEW',records=200,unique_control_ids=100,unique_Full_reference_ids=100,old_records_exact=184,complete_scenes=10,display_cells=cells,arithmetic=arithmetic,actual_root_adoption_refusals=refusals,prepared_seal_unchanged=True,checkpoint_config_root_valid_contracts_checked=200,native_tolerance=1e-12,native_shared_views_including_counts_identical=200,other_variants_excluded=True,torch_imported='torch' in sys.modules,new_inference=0,new_fit=0,new_training=0,canonical_changes=False)
    b.write(b.H/'FINAL_CHECKS.json',result);print(json.dumps(result))

if __name__=='__main__':main()
