"""One reversible source-only refinement inside the private A80 candidate."""
from pathlib import Path
import ast,difflib,hashlib,json
H=Path(__file__).resolve().parent
guard='''def interpretation_guard(panels):
    expected={
        ('F Flip','native'):['lower','lower','higher'],
        ('F Flip','shared_calibration'):['lower','lower','higher'],
        ('F Flip','raw'):['lower','higher','higher'],
        ('FedSA','native'):['higher','higher','higher'],
        ('FedSA','shared_calibration'):['higher','higher','higher'],
        ('FedSA','raw'):['higher','lower','lower'],
    }
    selected=[p for p in panels if p['seed_count']==10]
    require(len(selected)==6 and {(p['attack'],p['view']) for p in selected}==set(expected),'Six ten-seed interpretation bindings')
    for p in selected:
        require(p['directions']==expected[(p['attack'],p['view'])],'Actual A80 signs must support interpretation')
    return selected

'''
prose='''**Prediction-rule-dependent evidence.** In the ten-seed paired means for non-IID F Flip, deleting A lowers ACC and AEOD but raises ASPD under native and shared calibration: the utility loss accompanies an improvement in one disparity measure and a deterioration in the other. Under raw prediction, the same deletion lowers ACC and raises both gaps. For non-IID FedSA, raw prediction instead improves in all three paired-mean metrics after deleting A (higher ACC and lower AEOD/ASPD), providing a counterexample to a universal benefit from A. Native and shared calibration yield a small mean ACC increase but larger gaps in that scene. These contrasts show that the observed utility–disparity trade-offs depend on the prediction rule. They describe ten-seed paired means, not all individual seeds or statistically significant effects, and do not establish that A is indispensable. The linked table and saved values retain the separate 9- and 6-seed sensitivity panels.

'''
changes=[]
for name in ('increment_body.py.txt','build_and_check.py'):
    p=H/name;old=p.read_bytes();s=old.decode()
    (H/(name+'.before_interpretation')).write_bytes(old)
    replacements=[
        ('def build():',guard+'def build():'),
        ('    def counts(mapping):', '    interpretation=interpretation_guard(directions)\n    def counts(mapping):'),
        ('**Interpretation and fixed sensitivity panels.**',prose+'**Interpretation and fixed sensitivity panels.**'),
        ('scope=[dict(source=', 'interpretation_ten_seed_bindings=interpretation,\n        scope=[dict(source='),
        ("    for fact in facts['scope']", "    require(interpretation_guard(facts['new_nonIID_direction_panels'])==facts['interpretation_ten_seed_bindings'],'Six actual ten-seed interpretation facts')\n    for fact in facts['scope']"),
        ('fixed_10_9_6_direction_panels_bound=18,','fixed_10_9_6_direction_panels_bound=18,explicit_ten_seed_interpretation_sign_bindings=6,'),
    ]
    for a,b in replacements:
        assert s.count(a)==1,(name,a,s.count(a));s=s.replace(a,b)
    compile(s,str(p),'exec')
    p.write_bytes(s.encode())
    changes.extend(difflib.unified_diff(old.decode().splitlines(True),s.splitlines(True),fromfile=name+'.before_interpretation',tofile=name))
(H/'INTERPRETATION_SOURCE_DIFF.patch').write_text(''.join(changes),encoding='utf-8')
source=H/'build_and_check.py';prior=H/'build_and_check.py.before_interpretation'
def helpers(p):
    txt=p.read_text(encoding='utf-8');tree=ast.parse(txt)
    return {n.name:(ast.dump(n,include_attributes=False),ast.get_source_segment(txt,n)) for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('file_sha','read','save','original_checks')}
assert helpers(source)==helpers(prior)
assert source.read_text(encoding='utf-8').endswith((H/'increment_body.py.txt').read_text(encoding='utf-8'))
report=dict(status='SOURCE_REFINEMENT_COMPILED_ORIGINAL_HELPERS_EXACT',original_four_helpers_AST_and_text_exact=True,template_matches_builder_suffix=True,actual_build_run=False,old_prepared_seal_is_historical=True,files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(H.iterdir()) if p.is_file() and p.name!='SOURCE_READY_SHA256.json'})
(H/'SOURCE_READY_SHA256.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'status':report['status'],'builder_sha256':report['files']['build_and_check.py'],'seal_sha256':hashlib.sha256((H/'SOURCE_READY_SHA256.json').read_bytes()).hexdigest()}))
