"""Stdlib metadata/source checks; never runs original transport or science."""
from pathlib import Path
import ast,copy,hashlib,json
V=Path(__file__).resolve().parent;H=V.parent
read=lambda p:json.loads(Path(p).read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 inverse=read(V/'INVERSE_EDITS.json')
 for rel,info in inverse.items():
  raw=(V/rel).read_bytes();s=raw.decode()
  for e in reversed(info['edits']):assert s.count(e['after'])==e['count'];s=s.replace(e['after'],e['before'])
  p=H/info['parent'];assert sha(p)==info['parent_sha256'] and s.encode()==p.read_bytes()
 for n in ('transport.py','contract.py'):assert (V/n).read_bytes()==(H/'saved_v2'/n).read_bytes()
 orig=(H/'saved_v2/transport_remote.py').read_bytes().decode();new=(V/'transport_remote.py').read_bytes().decode()
 def ids_assignment(s):return next(n for n in ast.parse(s).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='ids' for t in n.targets))
 original_ids=ids_assignment(orig);new_ids=ids_assignment(new);expected=read(H/'MANIFEST.json')['exact_ids'];ids=ast.literal_eval(new_ids.value)
 assert ids==expected and len(ids)==len(set(ids))==10
 assert new.replace(ast.get_source_segment(new,new_ids),ast.get_source_segment(orig,original_ids))==orig
 assert ast.dump(ast.parse(new.replace(ast.get_source_segment(new,new_ids),ast.get_source_segment(orig,original_ids))))==ast.dump(ast.parse(orig))
 original_gate=read(H/'runtime/TERMINAL_GATE.json');actual_linux_path=H/'actual/saved001/LINUX_SAVED_CHECK.json';actual_linux=read(actual_linux_path)
 assert sha(actual_linux_path)=='30423f89729798d4726490d05a8c4ed187c2aa826094ea0b15228c919c1894c1'
 gate=dict(status=original_gate['gate_status'],package_sha256=original_gate['package_sha256'],receipts=[dict(id=i) for i in original_gate['exact_ids']])
 gate_hash=original_gate['gate_sha256'];linux_hash=sha(actual_linux_path)
 # Execute only the original plain metadata assertion nodes, never file reads/archive body.
 assertion_nodes=[n for n in ast.parse(new).body if isinstance(n,ast.Assert) and n.lineno in (9,10,27,28,29,30,31,32)]
 assert len(assertion_nodes)==8
 def validate(g,l,gh,lh):
  assert gh=='6bdf450e37432f8071b4ca0fc574edc22441ebb26c6ff7d29453868913953fc6' and lh=='30423f89729798d4726490d05a8c4ed187c2aa826094ea0b15228c919c1894c1'
  ns=dict(gate=g,linux=l,ids=ids);exec(compile(ast.Module(body=assertion_nodes,type_ignores=[]),'<original metadata asserts only>','exec'),ns)
 validate(gate,actual_linux,gate_hash,linux_hash)
 refusals={}
 cases=[('old_FFlip_scope',lambda g,l,x:g['receipts'][0].__setitem__('id',ast.literal_eval(original_ids.value)[0])),('missing_ID',lambda g,l,x:g['receipts'].pop()),('Linux_wrong_order',lambda g,l,x:l['records'].reverse()),('wrong_gate_sha',lambda g,l,x:x.__setitem__(0,'0'*64)),('wrong_linux_sha',lambda g,l,x:x.__setitem__(1,'0'*64)),('whole_not_exact',lambda g,l,x:l['records'][0].__setitem__('cached_root_fit_exact',False)),('wrong_inner_science',lambda g,l,x:l.__setitem__('original_check_saved_sha256','0'*64))]
 for name,edit in cases:
  g,l,x=copy.deepcopy((gate,actual_linux,[gate_hash,linux_hash]));edit(g,l,x)
  try:validate(g,l,*x)
  except AssertionError:refusals[name]=True
  else:raise AssertionError('Refusal not enforced: '+name)
 compiled=[]
 for q in V.glob('*.py'):compile(q.read_bytes(),str(q),'exec');compiled.append(q.name)
 failure=H/'actual/saved001/TRANSPORT_SSH_EXIT.json';assert read(failure)['exit']==1
 empty=Path('F:/YananResearchStorage/GuardFed/fl_FedSA10_three_view_prepared_20261011/attempt001/flgmm10_saved_arrays_and_receipts.zip');assert empty.is_file() and empty.stat().st_size==0
 assert not (H/'actual/saved002/TRANSPORT_COMMAND.json').exists()
 result=dict(status='TRANSPORT_V2_LITERAL_AND_ACTUAL_METADATA_SOURCE_CHECK_PASS_NOT_EXECUTED',remote_changed_span='only ids literal',raw_inverse_files=len(inverse),original_transport_client_raw_exact=True,original_contract_raw_exact=True,original_remote_body_other_than_ID_literal_raw_and_AST_exact=True,actual_gate_fields_source='runtime/TERMINAL_GATE.json (observed actual GATE fields, not newly downloaded full GATE)',actual_gate_summary_sha256=sha(H/'runtime/TERMINAL_GATE.json'),actual_gate_sha256=gate_hash,actual_Linux_proof_sha256=linux_hash,metadata_assertions_original=8,metadata_positive_actual_evidence=True,metadata_refusals=refusals,compiled=compiled,original_failure_exit1_preserved=True,F_attempt001_zero_byte_preserved=True,Windows_original_zero_fit_functions_untouched=True,new_SSH=0,new_transport=0,new_fit=0,new_CNN=0,root_adoption=False)
 with (V/'SOURCE_CHECK.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps(result))
if __name__=='__main__':main()
