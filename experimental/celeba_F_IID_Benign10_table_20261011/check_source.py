"""Source/metadata checks only. Does not invoke builder, arithmetic, fit or F reads."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import binding

H=Path(__file__).resolve().parent;R=H.parents[1]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    pins=json.loads((H/'SOURCE_INPUTS.json').read_bytes())
    old=(R/pins['original_builder']).read_text('utf8');new=(H/'build.py').read_text('utf8')
    segment=lambda s:s[s.index('    for rid in ids:'):s.index('    records=[fulls[k]')]
    mapped=segment(new).replace('minus_F','minus_A').replace("ablation_component']=='F'","ablation_component']=='A'")
    extra="        need(full['seed']==row['seed'] and full['distribution']==row['distribution']=='IID' and full['attack']==row['attack']=='Benign' and row['actual_alpha']==5000, 'Exact paired seed/IID Benign/alpha5000 required')\n"
    assert mapped.count(extra)==1;mapped=mapped.replace(extra,'')
    mapped=mapped.replace(",record_source_root=index['record_source_roots'][rid]",'')
    assert mapped==segment(old),'Original per-record identity loop changed beyond variant/extra exact-scope and provenance fields'
    numeric_original=(R/'tmp/celeba_mechanism_C_three_view_table_prepare_20261009/verify_numeric.py').read_text('utf8')
    numeric=(H/'verify_numeric.py').read_text('utf8')
    def loops(text):
        fn=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='verify')
        return [ast.dump(n,include_attributes=False) for n in fn.body if isinstance(n,ast.For)]
    assert loops(numeric)==loops(numeric_original),'Original fsum/sampleSD/count loops changed'
    for key in ('evidence','C_panels','receipt_identity_source','canonical_source','full_record_source'):
        path=R/pins[key];assert sha(path)==pins['files'][pins[key]]['sha256']
    funcs={}
    for key,names in [('evidence',('statistic','summarize')),('C_panels',('panels',)),('receipt_identity_source',('receipt_identity','normalized')),('full_record_source',('full_record',))]:
        text=(R/pins[key]).read_text('utf8')
        for n in ast.parse(text).body:
            if isinstance(n,ast.FunctionDef) and n.name in names:funcs[key+':'+n.name]=hashlib.sha256(ast.get_source_segment(text,n).encode()).hexdigest()
    # Symbolic metadata only: these are fixtures, never adopted results.
    prior_ids=[f'fixture_prior_{i}' for i in range(300)]+binding.IDS[:4]
    part=lambda ids:dict(new_ids=list(ids),new_records=[{'id':i} for i in ids],new_bindings={i:{} for i in ids},new_binding_files={i:{} for i in ids},new_artifacts={i:{} for i in ids})
    prior=part(binding.IDS[:4]);prior['all_ids']=prior_ids
    index=part(binding.IDS[4:]);index.update(all_ids=prior_ids+binding.IDS[4:],prior_index_sha256=binding.PARENT_INDEX_SHA,prior_adoption_sha256=binding.PARENT_ROOT_SHA)
    root=dict(prior_accepted=304,new_accepted=6,cumulative_accepted=310,status='ROOT_FIXTURE_ADOPTED',original304_unchanged=True,accepted_new_ids=binding.IDS[4:],native_max_abs_difference=0,Full_inference=0,new_CNN=0,new_training=0,test=False)
    native=dict(root_adopted=True,status='ROOT_ORIGINAL_STRICT_DELTA_ARCHIVE_AND_OFFSERVER_PASS',total_new_strict_and_offserver=312,test=False)
    results={}
    for case in ('valid_native312','native309','replay311','wrong_variant','wrong_seed','duplicate','old_prefix_changed','extra_binding','not_adopted','test_enabled','nonzero_native_difference'):
        r,i,p,n=copy.deepcopy((root,index,prior,native))
        if case=='native309':n['total_new_strict_and_offserver']=309
        if case=='replay311':r['cumulative_accepted']=311
        if case=='wrong_variant':r['accepted_new_ids'][0]=r['accepted_new_ids'][0].replace('minus_F','minus_V')
        if case=='wrong_seed':r['accepted_new_ids'][0]=r['accepted_new_ids'][0].replace('91005','91004')
        if case=='duplicate':i['all_ids'][-1]=i['all_ids'][-2]
        if case=='old_prefix_changed':i['all_ids'][0]='changed'
        if case=='extra_binding':i['new_bindings']['future']={}
        if case=='not_adopted':r['status']='PENDING'
        if case=='test_enabled':r['test']=True
        if case=='nonzero_native_difference':r['native_max_abs_difference']=1e-15
        try:binding.validate_scope(r,i,p,n);accepted=True
        except ValueError:accepted=False
        assert accepted==(case=='valid_native312'),case
        results[case]='ACCEPT' if accepted else 'REJECT'
    compiled=[]
    for p in H.glob('*.py'):compile(p.read_text('utf8'),str(p),'exec');compiled.append(p.name)
    print(json.dumps(dict(status='PASS_SOURCE_ONLY_ORIGINAL_STATISTICS_AND_METADATA_SCOPE',compiled=compiled,original_per_record_loop_inverse_exact=True,added_per_record_guards='Explicit matchingseed/IID Benign/alpha5000; provenance records actual304 or310 source root',original_fsum_SD_and_count_loops_AST_exact=True,original_source_functions=funcs,metadata_fixtures=results,numeric_builder_calls=0,scientific_verifier_calls=0,fit=0,CNN=0,F_bulk_reads=0),indent=2))

if __name__=='__main__':main()
