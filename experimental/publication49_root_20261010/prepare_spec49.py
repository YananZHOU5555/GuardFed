"""Reuse original48 selection/hash/receipt loop; only closed49 metadata is external."""
from pathlib import Path
import argparse,ast,importlib.util,json,sys
sys.dont_write_bytecode=True
from publish_increment49 import ROOT,H,HERE,OLD,PARENT,BRANCH,ACCEPTED,SCOPE,ROLES,FIXED_BINDINGS,REQUIRED_FACTS,base,configure,bind_snapshot
raw=(OLD/'prepare_spec.py').read_bytes();assert H(raw)=='feaebb0a47fbd8359291ae5f6ff3ab13e3ccdbbc286c64b6b7d9c29e39281962'
text=raw.decode('utf8');tree=ast.parse(text)
build_source=ast.get_source_segment(text,next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='build'))
TRAIN='docs/server_deployment_20260923/training_20260923/'
MUTABLE=[TRAIN+'TRAINING_STATE.json',TRAIN+'RUNNING.md',TRAIN+'REBUTTAL_COMPLETION_20261009.md',TRAIN+'server_reactivation_20261009/MONITOR_HANDOFF.md','docs/返修实验总览.md','tmp/update_reactivation_state_20261009.py','tmp/update_completion_current_20261009.py','tmp/update_overview_closure100_root_20261009.py']
PARENT_RECEIPT=Path('F:/YananResearchStorage/GuardFed/git_publication/increment48/stage001/STAGE_RECEIPT.json')
PARENT_RECEIPT_SHA='dbc5fc45ad215725a3efddf645c56571d5f0bf27c0feae4ac5d65ecefd9d5d26'
FIXED_SEALS={
 'tmp/celeba_gradient64_delta_after10_20261010/DELIVERY_FILES_SHA256.json':'c244607adb3ed5686ebd57196e2e6d759b875df410044b8d28e128c97687bfd3',
 'tmp/celeba_flgmm_fullcoverage_delta_after32_20261010/DELIVERY_FILES_SHA256.json':'207f662ee41db01d75cd96a3b2ed57f84b212c0ce04ecfc34c06ae48fb99357c',
 'tmp/celeba_hybrid32_final_collection_20261010/FILES_SHA256.json':'4370962e33b24e7fe9c5ffb461b1a99adc39621bc46efaaf25deb28c64b26e59',
 'tmp/celeba_hybrid_fullcoverage_implementation_v3_20261010/FILES_SHA256.json':'bce84aa075a242ec89c2fd016dbac8fb7dea4a1b369074a30fefd5ce6e61d7af'}


def parent_proof(number):
    return f'tmp/publication{number}_root'+('_v2' if number==48 else '')+'_20261010/REMOTE_VERIFICATION.json'


def seal_files(seal):
    d=json.loads(seal.read_bytes())
    if 'files' in d:return d['files']
    members=d['members'];assert isinstance(members,list)
    assert len({x['path'] for x in members})==len(members)
    return {x['path']:{'sha256':x['sha256'],'bytes':x['size']} for x in members}


replacements=[
 ("configure({'native':220,'three_view':220})","configure(ACCEPTED)"),
 ("Path('F:/YananResearchStorage/GuardFed/git_publication/increment47/stage001/STAGE_RECEIPT.json')","PARENT_RECEIPT"),
 ("assert H(receipt.read_bytes())=='466f5dfc114d3462a0b89fb7ef6786769f0af0bec2010b809d34970aaeb2c79e'","assert H(receipt.read_bytes())==PARENT_RECEIPT_SHA"),
 ("(47,'466f5dfc114d3462a0b89fb7ef6786769f0af0bec2010b809d34970aaeb2c79e')]:","(47,'466f5dfc114d3462a0b89fb7ef6786769f0af0bec2010b809d34970aaeb2c79e'),(48,PARENT_RECEIPT_SHA)]:"),
 ("f'tmp/publication{number}_root_20261010/REMOTE_VERIFICATION.json'","parent_proof(number)"),
 (next(line.strip() for line in build_source.splitlines() if line.strip().startswith('for directory,name in (')),'for directory,name in SEALED:'),
 ("json.loads(seal.read_bytes())['files'].items()","seal_files(seal).items()"),
 ("scope='CLOSED220_REPLAY220_GRADIENT10_LOGO100_A20_NO_BULK',accepted={'native':220,'three_view':220}","scope=SCOPE,accepted=ACCEPTED"),
 ("Only new native2, replay A8, gradient5, root-adopted LoGo100 and A20 table; future full reply/ten-method table only actual root extras; bulk excluded","Actual native236/replay236/FL38/gradient18/Hybrid32 plus bound source, seven canaries and96 startup; compact only, no formal96 accepted results")]
node=next(n for n in ast.walk(ast.parse(build_source)) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='paths' for t in n.targets))
replacements.append((ast.get_source_segment(build_source,node),'paths=ROLE_PATHS'))
for before,after in replacements:
    assert build_source.count(before)==1,before
    build_source=build_source.replace(before,after)


def build(closed):
    assert closed['status']=='ROOT_CLOSED49_INPUTS_READY_NOT_PUBLISHED'
    assert closed['parent']==PARENT and closed['accepted']==ACCEPTED
    bindings=closed['bindings'];assert set(bindings)==ROLES
    for role,pin in bindings.items():
        assert pin and pin.get('path') and pin.get('sha256'),'Actual binding missing: '+role
        if role in FIXED_BINDINGS:assert (pin['path'],pin['sha256'])==FIXED_BINDINGS[role],role
    frozen=closed['frozen_mutable'];assert set(frozen)==set(MUTABLE)
    for n,pin in frozen.items():
        assert pin and pin.get('sha256') and isinstance(pin.get('bytes'),int),'Shared entries not frozen: '+n
    assert bindings['current_state']['path']==MUTABLE[0] and bindings['current_state']['sha256']==frozen[MUTABLE[0]]['sha256']
    source_root=bind_snapshot(closed['source_snapshot'])
    seals=dict(FIXED_SEALS);replay=closed['replay_source_seal'];assert replay and replay.get('path') and replay.get('sha256'),'Actual A36 source seal missing'
    seals[replay['path']]=replay['sha256']
    pins={n:s for n,s in FIXED_BINDINGS.values()};pins.update({x['path']:x['sha256'] for x in bindings.values()});pins.update(seals);pins.update({n:v['sha256'] for n,v in frozen.items()})
    extras=closed['extra_paths'];assert isinstance(extras,list) and extras and all(isinstance(n,str) for n in extras)
    extras=list(extras)+list(seals)
    for n,s in seals.items():
        p=source_root/n;assert H(p.read_bytes())==s
        for member in seal_files(p):
            if Path(member).suffix.lower() in {'.json','.py','.md','.csv','.patch','.sha256'} and '__pycache__' not in Path(member).parts:
                extras.append(p.parent.relative_to(source_root).as_posix()+'/'+member)
    explicit=[TRAIN+'publication_closed_increment48_verified_20261010.json',*[x['path'] for x in bindings.values()],parent_proof(48),'tmp/publication48_root_v2_20261010/COMMIT_RECEIPT.json']
    ns=dict(ROOT=source_root,H=H,PARENT=PARENT,BRANCH=BRANCH,ACCEPTED=ACCEPTED,SCOPE=SCOPE,Path=Path,json=json,base=base,configure=configure,
      PARENT_RECEIPT=PARENT_RECEIPT,PARENT_RECEIPT_SHA=PARENT_RECEIPT_SHA,parent_proof=parent_proof,seal_files=seal_files,
      MUTABLE=MUTABLE,EXPLICIT=explicit,PINS=pins,SEALED=[(str(Path(n).parent.as_posix()),Path(n).name) for n in seals],DIRECTORIES=[],ROLE_PATHS={r:p['path'] for r,p in bindings.items()})
    for n,pin in frozen.items():assert len((source_root/n).read_bytes())==pin['bytes'],n
    exec(compile(build_source,str(HERE/'prepare_spec49.py')+':original48_build_metadata','exec'),ns)
    d=ns['build'](extras);d['source_snapshot']=closed['source_snapshot']
    return d


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--closed-inputs',type=Path,required=True);p.add_argument('--sha256',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    assert H(a.closed_inputs.read_bytes())==a.sha256
    assert a.out.resolve().is_relative_to(HERE) and a.out.name=='ACTUAL_SPEC.json'
    d=build(json.loads(a.closed_inputs.read_bytes()))
    with a.out.open('x',encoding='utf8') as f:json.dump(d,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps(dict(path=str(a.out),sha256=H(a.out.read_bytes()),files=len(d['files']),bytes=sum(e['bytes'] for e in d['files']))))
