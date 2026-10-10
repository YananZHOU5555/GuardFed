"""Same approved image-ID population rule, applied to existing accepted ID arrays."""
import argparse, ast, hashlib
from pathlib import Path
from metadata import HERE, SCREEN, bulk_path, digest, read, require, write, verify_sources

ORIGINAL=HERE.parent/'celeba_logofair_population_proposal_20261009/prepare_population.py'

def original_functions():
    # Load only these three pure ID functions: never run original main or read labels/scores.
    import numpy as np
    source=ORIGINAL.read_text(encoding='utf8');tree=ast.parse(source)
    namespace=dict(np=np,hashlib=hashlib,DOMAIN=bytes.fromhex(read(SCREEN/'mapping_metadata.json')['domain_hex']))
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('arrsha','cohorts','check_mapping')]
    require(len(nodes)==3,'Original ID mapping functions missing')
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(ORIGINAL),'exec'),namespace)
    return namespace

def verify_mapping(path,root_sha,valid_sha):
    f=original_functions();np=f['np']
    with np.load(path,allow_pickle=False) as z:m={k:z[k].copy() for k in z.files}
    f['check_mapping'](m,root_sha,valid_sha)

def prepare(seed,id_arrays,out):
    verify_sources();require(seed in range(91002,91011),'Seed91001 mapping remains unchanged')
    rows=[r for r in read(HERE/'CACHE_IDENTITIES100.json')['references'] if r['seed']==seed]
    require(len(rows)==10,'Actual seed inventory incomplete')
    arrays,_=bulk_path(id_arrays,0);row=next((r for r in rows if r['accepted_ID_arrays']['sha256']==digest(arrays)),None)
    require(row is not None,'ID arrays must match one actual accepted FedAvg receipt')
    out,_=bulk_path(out,4*1024*1024);require(not out.exists(),'No overwrite or partial retry')
    f=original_functions();np=f['np']
    with np.load(arrays,allow_pickle=False) as z:
        root=z['root_image_ids'].copy();valid=z['valid_image_ids'].copy()
    mapping=dict(root_image_id=root,valid_image_id=valid,root_client_id=f['cohorts'](root),valid_client_id=f['cohorts'](valid))
    f['check_mapping'](mapping,row['root_image_ids_sha256'],row['valid_image_ids_sha256'])
    out.mkdir(parents=True);np.savez_compressed(out/'mapping.npz',**mapping)
    meta=read(SCREEN/'mapping_metadata.json')
    meta.update(status='PREPARED_NOT_APPROVED',approved=False,execution_authorized=False,seed=seed,
        root_image_ids_sha256=row['root_image_ids_sha256'],valid_image_ids_sha256=row['valid_image_ids_sha256'],
        mapping_sha256=digest(out/'mapping.npz'),approval_scope='Pending actual100 binding; same image-ID rule, no population search',
        accepted_id_array_sha256=digest(arrays),decoded_fields=['root_image_ids','valid_image_ids'],
        valid_labels_or_scores_decoded=False,root_labels_decoded=False)
    write(out/'mapping_metadata.json',meta)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--seed',type=int,required=True);p.add_argument('--id-arrays',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();prepare(a.seed,a.id_arrays,a.out)
