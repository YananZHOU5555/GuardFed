"""Pure JSON identity projection; the original full inspection is never rewritten."""
from pathlib import Path
import hashlib,json,re

read=lambda p:json.loads(Path(p).read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
FIXED=[f'minus_F_non-IID_F Flip_seed{s}' for s in range(91001,91011)]


def record_tokens(payload):
    text=payload.decode('utf8');decoder=json.JSONDecoder()
    start=re.search(r'"records"\s*:\s*\[',text);assert start
    pos=start.end();tokens=[]
    while True:
        while text[pos].isspace() or text[pos]==',':pos+=1
        if text[pos]==']':break
        row,end=decoder.raw_decode(text,pos)
        assert isinstance(row,dict) and isinstance(row['id'],str)
        tokens.append((row,text[pos:end]));pos=end
    assert [r for r,t in tokens]==json.loads(payload)['records']
    assert len({r['id'] for r,t in tokens})==len(tokens)
    return tokens


def derive_scope(old_bytes,raw_bytes,fixed=FIXED):
    old=json.loads(old_bytes);raw=json.loads(raw_bytes)
    assert fixed==FIXED and not raw['invalid']
    old_tokens=record_tokens(old_bytes);raw_tokens=record_tokens(raw_bytes)
    rows={r['id']:(r,t) for r,t in raw_tokens}
    prior=old['accepted_new_ids'];observed=raw['accepted_new_ids']
    assert len(prior)==len(set(prior))==old['new_count']==360
    assert len(old_tokens)==460 and len(old['accepted_reused_ids'])==100
    assert len(observed)==len(set(observed))==raw['new_count']>=370
    assert set(prior)<=set(observed) and not set(prior)&set(fixed)
    assert set(fixed)<=set(observed) and raw['accepted_reused_ids']==old['accepted_reused_ids']
    assert set(rows)==set(observed)|set(raw['accepted_reused_ids'])
    for row,token in old_tokens:assert rows[row['id']]==(row,token),row['id']
    selected=old_tokens+[rows[i] for i in fixed]
    assert len(selected)==470 and len({r['id'] for r,t in selected})==470
    pending=[i for i in observed if i not in set(prior)|set(fixed)]
    return dict(status='ACCEPTED_NATIVE_SCOPE_METADATA_NOT_AN_INSPECTION',
        accepted_new_ids=prior+fixed,accepted_reused_ids=old['accepted_reused_ids'],
        new_count=370,reused_count=100,records=[r for r,t in selected],
        observed_native_count=raw['new_count'],pending_closed_ids=pending,
        row_token_sha256={r['id']:hashlib.sha256(t.encode('utf8')).hexdigest() for r,t in selected},
        old460_row_tokens_and_order_exact=True,Full100_references_exact=True,
        original_full_inspection_retained=True,root_adopted=False)


def derive_bound_scope(e,p):
    for prefix,source in [('native_inspection',e),('native360_inspection',p)]:
        assert sha(source[prefix+'_path'])==source[prefix+'_sha256']
    assert sha(e['native_root_path'])==e['native_root_sha256']
    proof=read(e['native_root_path'])
    assert proof['root_adopted'] is True and proof['test'] is False
    assert proof['total_new_strict_and_offserver']==e['root_native_total']==370
    assert proof['new_ids']==FIXED==p['candidate_ids']
    assert proof['inspection_sha256']==e['native_inspection_sha256']
    assert proof['previous_ledger_sha256']==p['native360_ledger_sha256']
    scope=derive_scope(Path(p['native360_inspection_path']).read_bytes(),Path(e['native_inspection_path']).read_bytes())
    scope.update(raw_inspection_path=e['native_inspection_path'],raw_inspection_sha256=e['native_inspection_sha256'],
        parent_inspection_path=p['native360_inspection_path'],parent_inspection_sha256=p['native360_inspection_sha256'],
        native_root_path=e['native_root_path'],native_root_sha256=e['native_root_sha256'])
    return scope


def load_scope(e,p):
    assert sha(e['accepted_native_scope_path'])==e['accepted_native_scope_sha256']
    scope=read(e['accepted_native_scope_path'])
    assert scope==derive_bound_scope(e,p)
    for row,token in record_tokens(Path(e['accepted_native_scope_path']).read_bytes()):
        assert hashlib.sha256(token.encode('utf8')).hexdigest()==scope['row_token_sha256'][row['id']]
    return scope


def write_scope(e,p,path):
    scope=derive_bound_scope(e,p)
    old=record_tokens(Path(p['native360_inspection_path']).read_bytes())
    raw=dict((r['id'],t) for r,t in record_tokens(Path(e['native_inspection_path']).read_bytes()))
    tokens=[t for r,t in old]+[raw[i] for i in FIXED]
    header={k:v for k,v in scope.items() if k!='records'}
    text=json.dumps(header,ensure_ascii=False,indent=2,allow_nan=False)
    payload=(text[:-2]+',\n  "records": [\n'+',\n'.join(tokens)+'\n  ]\n}\n').encode('utf8')
    assert json.loads(payload)==scope
    with Path(path).open('xb') as f:f.write(payload)
    e['accepted_native_scope_path']=str(path)
    e['accepted_native_scope_sha256']=sha(path)
    return load_scope(e,p)
