"""Compile, metadata rejection fixtures and unchanged original loop AST checks only."""
from pathlib import Path
import ast,copy,hashlib,json
from input_contract import IDS,PACKAGE,PACKAGE_SHA,validate_binding,validate_new_rows
H=Path(__file__).resolve().parent;R=H.parents[1]

def loop(path,needle):
    src=path.read_text(encoding='utf8');tree=ast.parse(src)
    matches=[n for n in ast.walk(tree) if isinstance(n,ast.For) and needle in (ast.get_source_segment(src,n) or '')]
    return min(matches,key=lambda n:n.end_lineno-n.lineno)

def main():
    for name in ('build.py','verify.py','input_contract.py','check_source.py'):
        compile((H/name).read_text(encoding='utf8'),str(H/name),'exec')
    old=R/'tmp/flgmm_six_scene_table_20261011'
    comparisons={}
    for name,needle,label in [('build.py',"a=accepted[rid];row=byid[rid]",'per_record_identity'),('build.py',"rows=sorted([r for r in groups",'new_scene_statistical_body'),('verify.py',"receipt=read(r['receipt_path'])",'new_receipt_count_body'),('verify.py',"vals=[r['views'][pan['view']][metric]",'fsum_display_metric_body')]:
        a=loop(old/name,needle);b=loop(H/name,needle)
        # Iteration scope changes to new10/newscene only; the original body remains exact.
        da=ast.dump(ast.Module(body=a.body,type_ignores=[]),include_attributes=False)
        db=ast.dump(ast.Module(body=b.body,type_ignores=[]),include_attributes=False)
        assert da==db,label
        comparisons[label]={'original_body_AST_exact':True,'AST_sha256':hashlib.sha256(da.encode()).hexdigest()}
    fixture={'root_adopted':True,'root71':PACKAGE+'/ROOT_SCIENTIFIC_ADOPTION.json','root71_sha256':'a'*64,'candidate':PACKAGE,'candidate_seal_sha256':PACKAGE_SHA,'transport':'tmp/fixture/TRANSPORT_VERIFICATION.json','transport_sha256':'b'*64}
    rows=[dict(id=rid,method='FLGMM',distribution='non-IID',attack='F Flip',seed=seed,checkpoint_sha256='c'*64,array_sha256='d'*64,receipt_sha256='e'*64) for seed,rid in zip(range(91001,91011),IDS)]
    validate_binding(fixture);validate_new_rows(rows)
    invalid=[]
    for key,value in [('root_adopted',False),('root71_sha256',None),('candidate','tmp/other'),('transport','F:/bulk/TRANSPORT_VERIFICATION.json')]:
        item=copy.deepcopy(fixture);item[key]=value;invalid.append((lambda x:validate_binding(x),item,key))
    invalid.append((validate_new_rows,rows[:-1],'missing_seed'))
    for key,value in [('id',IDS[1]),('attack','FedSA'),('seed',91011),('checkpoint_sha256','invalid')]:
        item=copy.deepcopy(rows);item[0][key]=value;invalid.append((validate_new_rows,item,key))
    for fn,item,label in invalid:
        try:fn(item)
        except ValueError:pass
        else:raise AssertionError('Did not reject '+label)
    result={'status':'SOURCE_COMPILE_AST_AND_METADATA_FIXTURES_PASS','original_scientific_bodies':comparisons,'rejection_fixtures':len(invalid),'actual_binding_created':False,'numeric_builder_executed':False,'scientific_verifier_executed':False,'new_fit':0,'new_inference':0}
    (H/'SOURCE_CHECK.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps(result))
if __name__=='__main__':main()
