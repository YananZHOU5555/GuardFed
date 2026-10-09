"""Finite source/terminal guard checks only; no helper main, server or scientific runtime."""
from pathlib import Path
import ast,copy,json,hashlib
ROOT=Path(__file__).resolve().parents[2]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    names=['backup_mechanism_after71_root_20261009.py','adopt_mechanism_after71_root_20261009.py']
    trees={n:ast.parse((ROOT/'tmp'/n).read_text(encoding='utf-8')) for n in names}
    for n,t in trees.items():compile(t,n,'exec')
    terminal=[]
    for t in trees.values():
        f=next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name=='check_terminal')
        terminal.append(f)
    assert ast.dump(terminal[0],include_attributes=False)==ast.dump(terminal[1],include_attributes=False)
    namespace={};exec(compile(ast.Module(body=[terminal[0]],type_ignores=[]),'<pure terminal guard>','exec'),namespace)
    check=namespace['check_terminal']
    base=ROOT/'tmp/celeba_mechanism_valid_incremental_after71_20261009'
    expected=json.loads((base/'SCOPE.json').read_bytes())['selected_ids']
    live={'service':'guardfed_celeba_mechanism_valid_after71 EXITED','processes':[],'batch_failure':None,'batch_complete':{'status':'fixture_only'},'completed':[{'id':i} for i in expected]}
    check(live,expected)
    variants=[]
    for field,value in [('service','RUNNING'),('processes',[{'pid':1}]),('batch_failure',{'error':'fixture_only'}),('batch_complete',None)]:
        bad=copy.deepcopy(live);bad[field]=value;variants.append(bad)
    bad=copy.deepcopy(live);bad['completed'].pop();variants.append(bad)
    bad=copy.deepcopy(live);bad['completed'][0]['id']='wrong_id';variants.append(bad)
    bad=copy.deepcopy(live);bad['completed'][0]['id']=bad['completed'][1]['id'];variants.append(bad)
    for bad in variants:
        try:check(bad,expected)
        except AssertionError:pass
        else:raise AssertionError('Invalid terminal accepted')
    try:check(live,expected[:-1])
    except AssertionError:pass
    else:raise AssertionError('Invalid expected scope accepted')
    old=ast.parse((ROOT/'tmp/adopt_mechanism_next11_root_20261009.py').read_text(encoding='utf-8'))
    old_archive=next(n for n in old.body if isinstance(n,ast.With) and isinstance(n.items[0].context_expr,ast.Call) and ast.unparse(n.items[0].context_expr.func)=='tarfile.open')
    old_assert_nodes=[n.test for n in ast.walk(old_archive) if isinstance(n,ast.Assert)]
    class RebindCount(ast.NodeTransformer):
        changed=0
        def visit_Constant(self,node):
            if node.value==120:
                self.changed+=1;return ast.Name(id='expected_members',ctx=ast.Load())
            return node
    rebind=RebindCount()
    old_asserts=[ast.dump(rebind.visit(copy.deepcopy(n)),include_attributes=False) for n in old_assert_nodes]
    assert rebind.changed==1
    membership=next(n for n in trees[names[1]].body if isinstance(n,ast.FunctionDef) and n.name=='expected_archive_names')
    member_namespace={};exec(compile(ast.Module(body=[membership],type_ignores=[]),'<pure archive names>','exec'),member_namespace)
    archive_names=member_namespace['expected_archive_names'](expected,json.loads((base/'FILES_SHA256.json').read_bytes()),json.loads((base/'execution_candidate/EXECUTION_SOURCE_SHA256.json').read_bytes()))
    assert len(archive_names)==110 and 'backup_inventory.json' in archive_names
    assert len(archive_names-{'backup_inventory.json'})==109
    new_asserts=[ast.dump(n.test,include_attributes=False) for n in ast.walk(trees[names[1]]) if isinstance(n,ast.Assert)]
    assert all(a in new_asserts for a in old_asserts), 'Original archive/member/source guards changed'
    for expression in ["(proof['independent_metric_checks'],proof['independent_confusion_count_checks'],proof['prediction_rule_checks'])==(99,264,33)","row['native_max_abs_difference']==0 and set(row['views'])=={'native','raw','shared_calibration'}","row['checkpoint_sha256']==original[row['id']]['checkpoint']['sha256']"]:
        assert ast.dump(ast.parse(expression,mode='eval').body,include_attributes=False) in new_asserts
    assignments=[n for n in ast.walk(trees[names[0]]) if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='code' for x in n.targets)]
    remote=assignments[0].value.left.value
    compile(remote%('/workspace/guardfed_checks/celeba_mechanism_valid_incremental_after71_20261009/execution_candidate','executionSHA','approvalSHA','scienceSHA',live['batch_complete'],expected),'<remote source only>','exec')
    result={'status':'NO_CNN_SOURCE_AND_TERMINAL_GUARDS_PASS','terminal_refusals':8,'original_archive_assertions_preserved_with_member_count_rebind':len(old_asserts),'member_count_rebinds':rebind.changed,'actual_archive_members':len(archive_names),'actual_content_members':len(archive_names)-1,'original_scientific_assertions_exact':3,'remote_source_compiles':True,'helper_mains_executed':False,'server_or_subprocess_executed':False,'CNN_or_scientific_runtime_imported':False,'helpers':{n:sha(ROOT/'tmp'/n) for n in names}}
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
