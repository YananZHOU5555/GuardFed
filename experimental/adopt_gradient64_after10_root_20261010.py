"""Reuse the accepted root checks with exact8/parent10 operational bindings."""
from pathlib import Path
import ast,hashlib,json,math,re
ROOT=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
parent=ROOT/'tmp/adopt_gradient64_after5_root_20261010.py'
assert sha(parent)=='07cdee493ddad8f74290d3e3f6122bdaf8baa6ac9c0a9378894a3cf72aa59908'
node=next(n for n in ast.parse(parent.read_text('utf8')).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='bindings' for t in n.targets))
bindings=ast.literal_eval(node.value)
directory='celeba_gradient64_delta_after10_20261010'
pins={
 '551bf7b32961843ad15fc535aeddf1115402d7d3a3d9d8642ed118d3f7b9d0ec':'c244607adb3ed5686ebd57196e2e6d759b875df410044b8d28e128c97687bfd3',
 'd6dfd04ded669b4ae43612e747d2b6da6248ca543b99c13d6fe4459b7d9c9e0f':'4faaae38e737d31c13f6a5e15e6d22c521e649893068836b087bf4e857b75b6d',
 'e040d742c082dbc65dbb8b7cc36055cf869b5e75950dd927d24a362471b0ff95':'99a07fefad41e0f441288f86c5335e175eb5992983fec047ab3a1c9f1eacea57',
 'b07ce31865d243d0ac7447cbaa860197046e33e5323c3f078b966a36e80970ac':'79c4f0dd11bd1902eef9b65739d736133914316266e91240b20f7746983ad2ba',
 '37f5afcc529c021ed20931c960ce2d07bf8b0954439ff997d5b5f5e47f2eabdc':'ac7c207bcb23111d03e23b19f85877c2bbd1a11780f1a0f4e518f48f88dc3736',
 '0c936b662f08cc7508874c828213c7bc4c3cfe7fca3ac5988646b5d5cc682b4e':'1e4717bd5f7f66569198ff01b0f2a4d50b8bccb6f6d8cb8e24776e69973918f8',
}
changes={
 'celeba_gradient64_delta_after5_20261010':directory,
 "handoff['accepted_before']==5":"handoff['accepted_before']==10",
 "handoff['new_strict_offserver_count']==5 and handoff['strict_offserver_cumulative']==10":"handoff['new_strict_offserver_count']==8 and handoff['strict_offserver_cumulative']==18",
 "len(set(handoff['accepted_job_ids']))==10":"len(set(handoff['accepted_job_ids']))==18",
 "off['accepted_total']==10":"off['accepted_total']==18",
 "len(index['files'])==142":"len(index['files'])==163",
 "member_proof['members_verified']==139":"member_proof['members_verified']==160",
 "tensor['models']==5":"tensor['models']==8",
 "len(off['records'])==5":"len(off['records'])==8",
 "sum(r['constant_negative_retained'] for r in off['records'])==3":"sum(r['constant_negative_retained'] for r in off['records'])==0",
 "==40 and sum(r['elements'] for r in tensor['records'])==467530":"==64 and sum(r['elements'] for r in tensor['records'])==748048",
 'ROOT_GRADIENT64_EXACT5_ORIGINAL_STRICT_OFFSERVER_ADOPTED':'ROOT_GRADIENT64_EXACT8_ORIGINAL_STRICT_OFFSERVER_ADOPTED',
 'accepted_before=5,accepted_new=5,accepted_total=10':'accepted_before=10,accepted_new=8,accepted_total=18',
 'archive_members_root_verified=139,raw_files_root_verified=142':'archive_members_root_verified=160,raw_files_root_verified=163',
 'Exact5 scientific acceptor':'Exact8 scientific acceptor',
 'accepted_total=10,new=5':'accepted_total=18,new=8',
}
for key,value in list(bindings.items()):
    for before,after in dict(pins,**changes).items():value=value.replace(before,after)
    bindings[key]=value
bindings["R(D/'COLLECTOR_RELEASE.json')['CPU106_released']"]="R(D/'COLLECTOR_RELEASE.json')['CPU110_released']"
bindings['CPU106_released=True']='CPU110_released=True'
original=ROOT/'tmp/adopt_gradient64_after1_root_20261010.py'
assert sha(original)=='2b014b5935bf738cbdf54ede895b8f1d8d5a4468257098e37154b81357dd9569'
source=original.read_text('utf8')
for key in bindings:assert source.count(key)==1,key
bound=re.sub('|'.join(re.escape(k) for k in sorted(bindings,key=len,reverse=True)),lambda m:bindings[m.group()],source)
ast.parse(bound)
assert not (ROOT/'tmp'/directory/'ROOT_ADOPTION_REVIEW.json').exists()
exec(compile(bound,str(original),'exec'),{'__file__':str(original),'__name__':'__main__','math':math})
