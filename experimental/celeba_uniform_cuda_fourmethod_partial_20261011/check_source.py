"""Compile and actual compact identity checks only; never call statistics/build/arrays."""
from pathlib import Path
import ast,json,hashlib,sys,copy
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;R=H.parents[1]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())
for p in H.glob('*.py'):compile(p.read_bytes(),str(p),'exec')
der=read(H/'STATS_SOURCE.json');assert sha(der['parent']['path'])==der['parent']['sha256']
expected=der['original_summaries']
for change in der['changes']:assert change['before'] in expected;expected=expected.replace(change['before'],change['after'])
stats=(H/'statistics_original.py').read_text('utf8');tree=ast.parse(stats)
def function(name):
 n=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
 return ''.join(stats.splitlines(True)[n.lineno-1:n.end_lineno])
assert function('summaries')==expected and hashlib.sha256(expected.encode()).hexdigest()==der['derived_summaries_sha256']
assert function('original_values_for')==der['original_display_function']
pins=read(H/'INPUT_PINS.json')
assert sha(R/pins['renderer']['path'])==pins['renderer']['sha256']
binding=read(H/'ROOT_BINDING.template.json');assert binding['root_execution_authorized'] is False and binding['root_source_review']==dict(path=None,sha256=None)
import build
_,metadata,docs,audits=build.verified_inputs(binding)
assert len(metadata)==400 and set(docs)==set(build.METHODS)
reject=0
bad=copy.deepcopy(binding);bad['roots']['GuardFed-AD2+']['sha256']='0'*64
try:build.verified_inputs(bad)
except ValueError:reject+=1
else:raise AssertionError('Changed Full proof accepted')
for pin in [dict(path='F:/YananResearchStorage/forbidden.json',sha256='a'*64),dict(path=None,sha256=None)]:
 try:build.local(pin)
 except ValueError:reject+=1
 else:raise AssertionError('Invalid/unbound file allowed')
bt=ast.parse((H/'build.py').read_bytes());fn=next(n for n in bt.body if isinstance(n,ast.FunctionDef) and n.name=='main')
gate=next(n.lineno for n in ast.walk(fn) if isinstance(n,ast.Constant) and n.value=='Statistics require separate explicit root authority')
summary=next(n.lineno for n in ast.walk(fn) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='summaries')
assert gate<summary and 'partial four-method uniformly-CUDA validation' in stats
oldmetrics_access=[n for n in ast.walk(bt) if isinstance(n,ast.Subscript) and isinstance(n.value,ast.Name) and n.value.id=='old' and isinstance(n.slice,ast.Constant) and n.slice.value=='metrics']
assert not oldmetrics_access
report=dict(status='SOURCE_AND_ACTUAL_ROOT_IDENTITY_PASS_STATISTICS_NOT_EXECUTED',compile=True,methods=4,identity_rows=400,old1100_used_for_metadata_only=True,all_root_audit_transport_SHAs_and_native_record_links_verified=True,all100_each_method_checkpoint_grid_exact=True,original_mean_stdev_math_preserved=True,display_function_normalized_text_exact=True,summaries_only_old_tenmethod_guard_removed=True,ddof=1,seeds=[10,9,6],cross_scene_seed_first_preserved=True,root_authority_gate_before_statistics=True,tamper_or_bulk_path_rejections=reject,mean_SD_computed=False,table_values_generated=False,F_access=False,SSH=False,arrays=False,fit=False,CNN=False)
with (H/'SOURCE_CHECK.json').open('x',encoding='utf8') as f:json.dump(report,f,indent=2);f.write('\n')
print(json.dumps(report))
