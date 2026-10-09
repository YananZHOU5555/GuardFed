"""Isolated original-screen acceptance process; no rewritten old result identities."""
import argparse
import json
from pathlib import Path
import sys

if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--release',type=Path,required=True);parser.add_argument('--id',required=True)
    args=parser.parse_args();sys.path.insert(0,str(args.release))
    from screen_common import local_identity,accepted,digest
    assert digest(args.release/'PACKAGE_SHA256.json')=='aec95ceb5e8c9b7aa9cec89e9d70d2700c2f242c29e918792088648d6269bad4'
    protocol,manifest=local_identity();item=next(r for r in manifest['jobs'] if r['id']==args.id)
    result=accepted(item,args.release/'runs'/args.id);assert result is not None
    print(json.dumps(dict(id=args.id,job_sha256=item['job_sha256'],checkpoint_sha256=digest(args.release/'runs'/args.id/'model.pt'),
                         seed=result['seed'],distribution=result['distribution'],attack=result['attack'],rounds=result['rounds'],
                         candidate=result['tuning_candidate'],metrics=result['metrics'],source_hashes=result['revision_job']['source_hashes'])))
