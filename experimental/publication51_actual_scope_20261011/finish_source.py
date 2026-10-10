"""Resume only the unwritten source tail after the recorded inverse-helper failure."""
from pathlib import Path
import ast, difflib, hashlib, json
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OLD=ROOT/'tmp/publication50_actual_scope_20261010'
PARENT='4dfc9403c182c8f192c374e396eb2f564970159c'
sha=lambda b:hashlib.sha256(b).hexdigest()
read=lambda p:json.loads(p.read_bytes())
def save(path,d):
    with path.open('x',encoding='utf-8',newline='\n') as f:f.write(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def replace(text,before,after,changes):
    assert text.count(before)==1,repr(before)
    changes.append(dict(before=before,after=after))
    return text.replace(before,after)
prepared=read(HERE/'PREPARED_MANIFEST.json')
SCOPE=prepared['scope'];accepted=prepared['accepted'];bindings=prepared['bindings']
facts={role:pin['expect'] for role,pin in bindings.items() if pin}
facts['current_state']={'/celeba_mechanism_v1/scientific_results_strictly_accepted':272,
                        '/celeba_mechanism_v1/three_view_new_models_accepted':260}
boundary={key:prepared[key] for key in ['FL47_new_three_view_root_accepted','whole_windows_array_block_pass',
    'single_record_diagnostic_records','single_record_diagnostic_scientific_acceptances',
    'platform_cause_established','final_partition_metadata_only']}
byname={e['source']:e for e in prepared['files']}
new_names=[e['source'] for e in read(ROOT/'tmp/guardfed_publication51_source_preparation_20261011/attempt_v2/DIAGNOSTIC_INPUT_SUPPLEMENT.json')['added_files']]
assert read(HERE/'PREPARATION_FAILURE.json')['publisher_written_before_failure'] is False
assert not (HERE/'publish_increment51.py').exists()
text=(HERE/'prepare_source.py').read_text(encoding='utf-8')
tree=ast.parse(text)
main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
start=next(i for i,n in enumerate(main.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='old_text' for t in n.targets))
exec(compile(ast.Module(body=main.body[start:],type_ignores=[]),str(HERE/'prepare_source.py'),'exec'),globals())
