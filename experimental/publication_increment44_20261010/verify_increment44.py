"""Original committed-blob verifier with Git44 paths and per-process Git options."""
from pathlib import Path
import argparse, ast, hashlib, json
from publish_increment44 import ROOT, CHECKOUT, BRANCH, ORIGIN, PARENT, git, storage, relative
p=ROOT/'tmp/publication_increment43_20261010/verify_increment43.py'
b=p.read_bytes(); assert hashlib.sha256(b).hexdigest()=='fedf45595ae97ea9dfb5f09461569b8bb835469d984e35e1951f40dee2b5a9a0'
tree=ast.parse(b)
body=ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='verify'],type_ignores=[])
assert len(body.body)==1
exec(compile(body,str(p),'exec'),globals())
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--receipt',type=Path,required=True);p.add_argument('--receipt-sha256',required=True)
    p.add_argument('--commit',required=True);p.add_argument('--remote',action='store_true');a=p.parse_args()
    assert len(a.commit)==40 and all(c in '0123456789abcdef' for c in a.commit)
    print(json.dumps(verify(a.receipt,a.receipt_sha256,a.commit,a.remote),indent=2))
