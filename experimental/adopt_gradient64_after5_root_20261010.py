"""Bind the new five saved-state records to the original gradient root checks."""
from pathlib import Path
import ast, hashlib, json, math, re
ROOT=Path(__file__).resolve().parents[1]
original=ROOT/'tmp/adopt_gradient64_after1_root_20261010.py'
assert hashlib.sha256(original.read_bytes()).hexdigest()=='2b014b5935bf738cbdf54ede895b8f1d8d5a4468257098e37154b81357dd9569'
source=original.read_text('utf8')
bindings={
 'celeba_gradient64_delta_after1_20261010':'celeba_gradient64_delta_after5_20261010',
 '833affd3beb8264c139683fc2ab80397b7ec5c7b31a7f70ff1cd4f15fed8f7ba':'551bf7b32961843ad15fc535aeddf1115402d7d3a3d9d8642ed118d3f7b9d0ec',
 'bfa52cdd3969c888bcc024a54f00219d6bc484c17fd69d7d3d8e3bb43fecb47e':'d6dfd04ded669b4ae43612e747d2b6da6248ca543b99c13d6fe4459b7d9c9e0f',
 'ca4a38076e5080d94069cdabd50a032d84a96a339914761ec568db44a43db978':'e040d742c082dbc65dbb8b7cc36055cf869b5e75950dd927d24a362471b0ff95',
 "handoff['accepted_before']==1":"handoff['accepted_before']==5",
 "handoff['new_strict_offserver_count']==4 and handoff['strict_offserver_cumulative']==5":"handoff['new_strict_offserver_count']==5 and handoff['strict_offserver_cumulative']==10",
 "len(set(handoff['accepted_job_ids']))==5":"len(set(handoff['accepted_job_ids']))==10",
 'b3250c5986728a734dfee38eceef63ab160e65222dc875e8a1f310fc6848e6dd':'b07ce31865d243d0ac7447cbaa860197046e33e5323c3f078b966a36e80970ac',
 "off['accepted_total']==5":"off['accepted_total']==10",
 '9996a0b647f24e63a708cdb7e0297e135831625b3fd3f33097a8e4fc966f48b2':'37f5afcc529c021ed20931c960ce2d07bf8b0954439ff997d5b5f5e47f2eabdc',
 "len(index['files'])==135":"len(index['files'])==142",
 'f1a2133af5b537246570c80d732bfdebc86037b7bc46d87ca6c55d2e070c8c10':'0c936b662f08cc7508874c828213c7bc4c3cfe7fca3ac5988646b5d5cc682b4e',
 "member_proof['members_verified']==132":"member_proof['members_verified']==139",
 "tensor['models']==4":"tensor['models']==5",
 "    assert row['rounds']==70 and row['metrics']==dict(accuracy=0.5166859616449389,aeod=0.,aspd=0.)\n    assert row['constant_negative_retained'] and row['data_contract']['image_data_contract']['evaluation_split']=='valid'":
 "    assert row['rounds']==70 and row=={r['id']:r for r in handoff['records']}[row['id']]\n    assert all(math.isfinite(x) and 0<=x<=1 for x in row['metrics'].values())\n    assert row['constant_negative_retained']==(row['metrics']==dict(accuracy=0.5166859616449389,aeod=0.,aspd=0.))\n    assert row['data_contract']['image_data_contract']['evaluation_split']=='valid'",
 "proof=dict(status='ROOT_GRADIENT64_EXACT4_ORIGINAL_STRICT_OFFSERVER_ADOPTED',":"assert len(off['records'])==5 and sum(r['constant_negative_retained'] for r in off['records'])==3\nassert {r['id'] for r in off['records']}==set(auth['authorized_ids'])\nassert sum(r['tensor_count'] for r in tensor['records'])==40 and sum(r['elements'] for r in tensor['records'])==467530\nassert {(r['id'],r['checkpoint_sha256']) for r in tensor['records']}=={(r['id'],r['checkpoint_sha256']) for r in off['records']}\nproof=dict(status='ROOT_GRADIENT64_EXACT5_ORIGINAL_STRICT_OFFSERVER_ADOPTED',",
 'accepted_before=1,accepted_new=4,accepted_total=5':'accepted_before=5,accepted_new=5,accepted_total=10',
 'archive_members_root_verified=132,raw_files_root_verified=135':'archive_members_root_verified=139,raw_files_root_verified=142',
 "constant_negative_ids=auth['authorized_ids']":"constant_negative_ids=[r['id'] for r in off['records'] if r['constant_negative_retained']]",
 'Exact4 scientific acceptor':'Exact5 scientific acceptor',
 'accepted_total=5,new=4':'accepted_total=10,new=5',
}
for key in bindings: assert source.count(key)==1,key
bound=re.sub('|'.join(re.escape(k) for k in sorted(bindings,key=len,reverse=True)),lambda m:bindings[m.group()],source)
ast.parse(bound)
assert not (ROOT/'tmp/celeba_gradient64_delta_after5_20261010/ROOT_ADOPTION_REVIEW.json').exists()
exec(compile(bound,str(original),'exec'),{'__file__':str(original),'__name__':'__main__','math':math})
