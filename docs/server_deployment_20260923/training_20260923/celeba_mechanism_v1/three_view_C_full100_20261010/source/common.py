"""Original record/statistics helpers, loaded without models, arrays or inference."""
import ast, hashlib, importlib.util, json, sys
from pathlib import Path
from types import SimpleNamespace

sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent
R=H.parents[1]
OLD=R/'docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_C_eight_scenes_20261010'
FULL=R/'outputs/guardfed_tables/celeba_nine_method_three_view_20261009'
VIEWS=('native','raw','shared_calibration')
METRICS=('accuracy_pct','aeod','aspd')
EXPECTED=[f'minus_C_non-IID_{a}_seed{s}' for a in ('S-DFA','Sp-DFA') for s in range(91001,91011)]
SCENES=[(d,a) for d in ('IID','non-IID') for a in ('Benign','F Flip','FedSA','S-DFA','Sp-DFA')]

def need(ok,message):
    if not ok:raise ValueError(message)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_bytes())
def write(p,value):
    with Path(p).open('x',encoding='utf8',newline='\n') as f:f.write(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def verify_inputs():
    need(not sys.flags.optimize,'Optimized Python forbidden')
    for name,pin in read(H/'INPUT_PINS.json')['files'].items():
        path=R/name;need(sha(path)==pin['sha256'] and path.stat().st_size==pin['bytes'],'Pinned input changed: '+name)
def scientific():
    ns=dict(need=need,VIEWS=VIEWS,hashlib=hashlib,json=json,FULL=FULL,sha=sha)
    specs=[('tmp/celeba_mechanism_three_view_paired71_20261009/inputs.py',('receipt_identity','normalized')),
           ('tmp/celeba_mechanism_valid_C_after70_20261010/bridge.py',('canonical',)),
           ('tmp/celeba_mechanism_three_view100_tables_20261009/build.py',('full_record',))]
    proof={}
    for name,names in specs:
        source=(R/name).read_text(encoding='utf8');tree=ast.parse(source)
        nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
        need(len(nodes)==len(names),'Original function missing')
        for n in nodes:proof[n.name]=hashlib.sha256(ast.get_source_segment(source,n).encode()).hexdigest()
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(R/name),'exec'),ns)
    return SimpleNamespace(**ns),proof
def record_spans(text):
    decoder=json.JSONDecoder();position=text.index('"records"');position=text.index('[',position)+1;spans=[]
    while True:
        while text[position].isspace() or text[position]==',':position+=1
        if text[position]==']':return spans
        _,end=decoder.raw_decode(text,position);spans.append(text[position:end]);position=end
def require_exact20(ids):
    need(ids==EXPECTED and len(ids)==len(set(ids))==20,'Only exact two new ten-seed C scenes')
def displayed_cells(text,panels):
    lines=[x for x in text.splitlines() if x.startswith('| ') and ' ± ' in x]
    rows=[r for p in panels for r in p['rows']];need(len(lines)==len(rows)==270,'Exact270 scene rows required')
    for line,row in zip(lines,rows):
        cells=line.strip('| ').split(' | ')
        for index,m in enumerate(METRICS,3):
            need(cells[index]==f"{row[m]['mean']:.{3 if m=='accuracy_pct' else 5}f} ± {row[m]['sample_sd_ddof1']:.{3 if m=='accuracy_pct' else 5}f}",'Display differs')
    return len(rows)*3
