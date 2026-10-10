"""Root actual source pinning and one explicit saved-array/root-fit check."""
from pathlib import Path
import ast,datetime,hashlib,json,subprocess
ROOT=Path(__file__).resolve().parents[1]
S=ROOT/'tmp/celeba_added_cnn_exact3_scientific_acceptance_20261010'
O=ROOT/'tmp/celeba_added_cnn_exact3_root_execution_20261010'
PY=ROOT/'tmp/celeba_baselines/remaining_20261009/group_a/.venv/Scripts/python.exe'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_bytes())
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def main():
    assert not (O/'SCIENCE_EXIT.json').exists(),'Preserve scientific check; no automatic retry'
    assert sha(S/'FILES_SHA256.json')=='0ab455572efd56afd216c22bf783ae0b0743414165e70aa47d1da6369d262c7e'
    seal=read(S/'FILES_SHA256.json')
    for rel,pin in seal['files'].items():
        p=S/rel;assert p.resolve().is_relative_to(S.resolve()) and sha(p)==pin['sha256'] and p.stat().st_size==pin['bytes'],rel
    checked=[]
    for pin in read(S/'SOURCE_PINS.json')['files']:
        p=ROOT/pin['path'];assert sha(p)==pin['sha256'] and p.stat().st_size==pin['bytes']
        text=p.read_text(encoding='utf-8');nodes={n.name:n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)}
        for name,fn in pin['functions'].items():assert hashlib.sha256(ast.get_source_segment(text,nodes[name]).encode()).hexdigest()==fn['sha256'],name
        checked.append({'path':pin['path'],'sha256':pin['sha256'],'functions_checked':list(pin['functions'])})
    transport=read(O/'TRANSPORT_VERIFICATION.json')
    assert transport['status']=='ROOT_EXACT3_F_TRANSPORT_ALL_MEMBERS_SHA_PASS_NOT_SCIENTIFIC_ACCEPTANCE'
    base=Path(transport['verified_extract'])
    for rel,pin in transport['members'].items():assert sha(base/rel)==pin['sha256'] and (base/rel).stat().st_size==pin['bytes']
    sourcecheck=dict(status='ROOT_ORIGINAL_EXACT3_OFFSERVER_SOURCE_AND_TRANSPORT_BOUND',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_seal_sha256=sha(S/'FILES_SHA256.json'),source_functions=checked,transport_sha256=sha(O/'TRANSPORT_VERIFICATION.json'),transport_archive_sha256=transport['archive_sha256'],new_CNN=0,new_training=0,test=False)
    save(O/'ROOT_OFFSERVER_SOURCE_REVIEW.json',sourcecheck)
    argv=[str(PY),'-B',str(S/'verify_offserver.py'),'--bundle',str(base/'bundle'),'--metadata-npz',str(base/'metadata.npz'),'--output',str(O/'ROOT_EXECUTED_SAVED_ARRAY_CHECK.json'),'--allow-original-cached-root-refit']
    save(O/'SCIENCE_COMMAND.json',dict(argv=argv,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_review_sha256=sha(O/'ROOT_OFFSERVER_SOURCE_REVIEW.json'),root_authorized_cached_root_refits=3,new_CNN=0,new_training=0,test=False,automatic_retry=False))
    c=subprocess.run(argv,capture_output=True,timeout=180)
    (O/'SCIENCE_STDOUT.txt').write_bytes(c.stdout);(O/'SCIENCE_STDERR.txt').write_bytes(c.stderr)
    save(O/'SCIENCE_EXIT.json',dict(exit=c.returncode,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),stdout_sha256=sha(O/'SCIENCE_STDOUT.txt'),stderr_sha256=sha(O/'SCIENCE_STDERR.txt'),no_auto_retry=True))
    if c.returncode:
        print(c.stderr.decode(errors='replace'));raise SystemExit(c.returncode)
    result=read(O/'ROOT_EXECUTED_SAVED_ARRAY_CHECK.json')
    assert result['status']=='EXACT3_OFFSERVER_SAVED_ARRAY_AND_ORIGINAL_ROOT_REFIT_PASS_PENDING_ROOT_ADOPTION' and len(result['records'])==3
    assert result['cached_root_refits']==3 and result['new_CNN']==result['new_training']==0 and not result['test']
    print(json.dumps({'status':result['status'],'records':3,'cached_root_refits':3,'native_max_differences':[r['native_comparison']['max_abs_difference'] for r in result['records']],'proof_sha256':sha(O/'ROOT_EXECUTED_SAVED_ARRAY_CHECK.json')}))

if __name__=='__main__':main()
