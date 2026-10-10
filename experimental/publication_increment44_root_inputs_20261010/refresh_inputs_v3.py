"""Same actual-input reader plus independently adopted FL22→28; no publication operation."""
from pathlib import Path
import argparse,hashlib,json,sys
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];H=lambda b:hashlib.sha256(b).hexdigest()
original=(HERE/'refresh_inputs_v2.py').read_text(encoding='utf8')
changes={"'/flgmm_fullcoverage_v2_20261009/new_accepted':22":"'/flgmm_fullcoverage_v2_20261009/new_accepted':28",'FL_new_accepted=22':'FL_new_accepted=28'}
text=original
for a,b in changes.items():
    assert text.count(a)==1;text=text.replace(a,b)
ns={'__name__':'actual_inputs_reader_reused','__file__':str(__file__)}
exec(compile(text,str(HERE/'refresh_inputs_v2.py')+':FL28_metadata_only','exec'),ns)
p=ns['p'];F='tmp/celeba_flgmm_fullcoverage_delta_after22_20261010'

def append_fl(d):
    files={e['source']:e for e in d['files']}
    def add(name,expected=None):
        _,b=p.source(name,[name])
        if expected:assert H(b)==expected,name
        files[name]=dict(source=name,sha256=H(b),bytes=len(b));return json.loads(b) if name.endswith('.json') else None
    seal=add(F+'/DELIVERY_FILES_SHA256.json','5ee1309eee278f191cef5e1a55d7fca4dc6c3336cdbf5f72cfd4c8c60f50e794');omitted=[]
    for name,e in seal['files'].items():
        b=(ROOT/F/name).read_bytes();assert H(b)==e['sha256'] and len(b)==e['bytes']
        if name.endswith('.txt') or name=='SERVER_GUIDE.md':omitted.append(dict(source=F+'/'+name,sha256=e['sha256'],reason='Console/guide remains in original delivery'))
        else:add(F+'/'+name,e['sha256'])
    root=add(F+'/ROOT_ADOPTION_REVIEW.json','e51007c549f8cc95970ce06cb61b4a477c11eff5d00e05a9aeac9ffb30dd449c')
    assert (root['accepted_before'],root['accepted_new'],root['accepted_total'])==(22,6,28)
    latest=add('tmp/celeba_flgmm_fullcoverage_incremental_20261009/LATEST_BACKUP.json')
    assert 28 in [latest.get('accepted'),latest.get('accepted_total')]
    add('tmp/adopt_FL96_after22_delta_root_20261010.py')
    for name in ['BACKUP_SHA256.json','MEMBERS.json','OFFSERVER_ACCEPTANCE.json','PARTIAL_ACCEPTANCE.json']:
        add(HERE.relative_to(ROOT).as_posix()+'/FL28_compact/'+name)
    d['FL28_adopted_binding']=dict(path=F+'/ROOT_ADOPTION_REVIEW.json',sha256=files[F+'/ROOT_ADOPTION_REVIEW.json']['sha256'],accepted_new=6,accepted_total=28,previous_root_sha256=root['previous_root_adoption_sha256'])
    d['FL28_omitted_console']=omitted
    d.update(files=sorted(files.values(),key=lambda e:e['source']),allowed_paths=sorted(files))
    assert sum(e['bytes'] for e in d['files'])<p.ns['LIMIT']
    return d

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--main-live',required=True);a.add_argument('--startup');a.add_argument('--startup-sha256')
    a.add_argument('--extra',action='append',default=[]);a.add_argument('--output',type=Path,required=True);x=a.parse_args()
    assert x.output.resolve().is_relative_to(HERE.resolve()) and not x.output.exists()
    try:
        d=append_fl(ns['prepare'](x.main_live,x.startup,x.startup_sha256,x.extra));p.ns['save'](x.output,d)
        print(json.dumps(dict(path=str(x.output),sha256=H(x.output.read_bytes()),files=len(d['files']),bytes=sum(e['bytes'] for e in d['files']),startup=d['optional_bindings']['logofair100_startup'])))
    except Exception as e:
        p.ns['save'](x.output.with_name(x.output.stem+'_FAILURE.json'),dict(error=repr(e),noGit=True,noSSH=True,no_fit=True));raise
