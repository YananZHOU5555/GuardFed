"""Thin source-bound41 transport reuse for actual C10 closure and adopted C80 only."""
from pathlib import Path
import hashlib,sys
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;R=H.parents[1]
source=R/'tmp/publication41_preparation_20261010/publish_increment41.py'
assert hashlib.sha256(source.read_bytes()).hexdigest()=='4532959c1282c23b2829da5683f36d2e3ae5667fabd0c6bf165f1e710fd674c8'
text=source.read_text('utf8')
for old,new in [('increment41','increment42'),('INCREMENT41','INCREMENT42'),('59ec6455c1402ff3bfdac454cbf8de9daf0d216d','b57ae3c07e1013c17820c2eb038d701a356b27c6'),('three_view=170','three_view=180'),("{'native','FL','Hybrid','state','live','previous_publication'}","{'C10','table','state','live','previous_publication'}"),('mechanism_three_view_offserver_verified=170','mechanism_three_view_offserver_verified=180'),('C70_table_included=False,C70_in_full_rebuttal=False','C80_table_included=True,C80_in_full_rebuttal=False'),('len(set(archives.values()))==3','len(set(archives.values()))==1')]:
    assert old in text;text=text.replace(old,new)
start=text.index('def evidence_guard(');stop=text.index('\ndef plan(',start)
text=text[:start]+text[stop+1:]
start=text.index("    evidence_guard(d['native']");stop=text.index("    prev=d['previous_publication']",start)
text=text[:start]+"    scope_guard(c,d,roots,pinned)\n"+text[stop:]
old="assert not any('three_view_C_' in x or 'rebuttal_integrated_C' in x for x in rel.parts)"
new="assert not any('rebuttal_integrated_C' in x or ('three_view_C_' in x and x!='three_view_C_eight_scenes_20261010') for x in rel.parts)"
assert old in text;text=text.replace(old,new)
import importlib.util
spec=importlib.util.spec_from_file_location('increment42_scope',H/'scope_guard.py');guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)
namespace=dict(__name__='sealed41_transport_rebound42',__file__=str(__file__),scope_guard=guard.check)
exec(compile(text,str(source)+':metadata42','exec'),namespace)
plan=namespace['plan']
if __name__=='__main__':namespace['main']()
