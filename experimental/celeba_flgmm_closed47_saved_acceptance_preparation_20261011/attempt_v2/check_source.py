"""Pure-stdlib source/metadata checks; never fit, import Torch or contact server."""
import argparse, ast, copy, difflib, hashlib, importlib.util, json
from pathlib import Path
import contract as k

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=k.HERE/'SOURCE_CHECK.json');args=ap.parse_args()
    k.need(args.output.resolve().parent==k.HERE and not args.output.exists(),'New compact check path inside prepared source only')
    compiled=[]
    for path in sorted(k.HERE.glob('*.py')):
        compile(path.read_text(encoding='utf-8'),str(path),'exec');compiled.append(path.name)
    spec=importlib.util.spec_from_file_location('_derive_only',k.HERE/'derive_sources.py')
    derive=importlib.util.module_from_spec(spec);spec.loader.exec_module(derive)
    generated={};derive.save=lambda name,value:generated.__setitem__(name,value)
    derive.main()
    for name,value in generated.items():
        if name.endswith('.patch'):
            k.need((k.HERE/name).read_text(encoding='utf-8')==value,'Derivation diff drift: '+name)
        else:k.need((k.HERE/name).read_text(encoding='utf-8')==value,'Source derivation drift: '+name)
    m=k.read(k.CANDIDATE/'MANIFEST.json')
    k.need(k.sha(k.CANDIDATE/'FILES_SHA256.json')==k.PACKAGE,'Candidate package differs')
    k.need(k.IDS==m['exact_ids'] and len(k.IDS)==len(set(k.IDS))==47,'Fixed manifest differs')
    source=k.CANDIDATE/'originals/saved_science.py';text=source.read_text(encoding='utf-8')
    k.need(k.sha(source)==k.SAVED_SOURCE_SHA,'Original whole checker differs')
    node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='check_saved')
    function_sha=hashlib.sha256(ast.get_source_segment(text,node).encode()).hexdigest()
    block=next(n for n in node.body if isinstance(n,ast.With))
    block_sha=hashlib.sha256(ast.dump(block,include_attributes=False).encode()).hexdigest()
    k.need(function_sha=='ed8dc5eb2332ef8df176166087a91fb5f2b5043cc0e4669f7c985ca5b7452cfc','Original whole function differs')
    k.need(block_sha=='170bed968fecb3cd477603132b4d185a64da6f7d35ab120776449be9aae6b0ec','Original array AST differs')
    old=k.ROOT/'tmp/transport_added_cnn_exact3_root_20261010.py'
    main_node=next(n for n in ast.parse(old.read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef) and n.name=='main')
    assignment=next(n for n in main_node.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='remote' for t in n.targets))
    old_remote=ast.literal_eval(assignment.value);new_remote=(k.HERE/'transport_remote.py').read_text(encoding='utf-8')
    # The original byte/hash collection, receipt binding, ZIP construction and
    # before/after source hash loops remain identical AST subtrees.
    old_nodes={ast.dump(n,include_attributes=False) for n in ast.walk(ast.parse(old_remote)) if isinstance(n,(ast.For,ast.With))}
    new_nodes={ast.dump(n,include_attributes=False) for n in ast.walk(ast.parse(new_remote)) if isinstance(n,(ast.For,ast.With))}
    retained=len(old_nodes&new_nodes);k.need(retained>=5,'Original archive/member loop drift')
    fixture={'status':'LINUX_ORIGINAL_FLGMM47_WHOLE_SAVED_CHECK_PASS_NOT_ROOT_ADOPTED','records':[{'id':i,'root_receipt_exact':True,'cached_root_fit_exact':True,'saved_predictions_metrics_counts_exact':True} for i in k.IDS],'original_check_saved_sha256':k.SAVED_SOURCE_SHA,'package_sha256':k.PACKAGE,'gate_result_sha256':'0'*64,'cached_root_refits':47,'new_CNN':0,'new_training':0,'test':False}
    k.linux_proof(fixture)
    refused=[]
    for name,change in [('partial46',lambda v:v['records'].pop()),('duplicateID',lambda v:v['records'][1].update(id=k.IDS[0])),('wrong_source',lambda v:v.update(original_check_saved_sha256='0'*64)),('false_whole',lambda v:v['records'][0].update(root_receipt_exact=False))]:
        v=copy.deepcopy(fixture);change(v)
        try:k.linux_proof(v)
        except ValueError:refused.append(name)
        else:raise AssertionError('Metadata refusal missing: '+name)
    expected={'bundle/GATE_RESULT.json','bundle/metadata_receipt.json','LINUX_SAVED_CHECK.json'}|{'bundle/'+i+'/'+n for i in k.IDS for n in ('receipt.json','validation_predictions.npz')}
    result={'status':'PREPARED_SOURCE_AND_SYNTHETIC_METADATA_CHECK_PASS_NOT_SCIENTIFIC_ACCEPTANCE','compiled':compiled,'exact_ids':47,'expected_archive_members':len(expected)+1,'original_check_saved_file_sha256':k.SAVED_SOURCE_SHA,'original_check_saved_function_sha256':function_sha,'exact_original_array_AST_sha256':block_sha,'original_archive_member_loop_AST_subtrees_retained':retained,'derived_remote_sources_exact':True,'synthetic_positive_fixture_only':True,'metadata_refusals':refused,'source_execution':False,'SSH':False,'CNN':0,'cached_root_refits':0,'transport':False,'scientific_acceptances':0,'root_adopted':False}
    k.save(args.output,result)
    print(json.dumps(result))

if __name__=='__main__':main()
