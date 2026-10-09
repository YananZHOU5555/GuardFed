"""Explicit metadata/path/count rebind of successful C-after36 preparation."""
from pathlib import Path
import ast,hashlib,json,re
H=Path(__file__).resolve().parent;R=H.parents[1];O=H.with_name('celeba_mechanism_valid_C_after36_20261010')
source=(O/'prepare.py').read_text(encoding='utf-8-sig')
selected=[f'minus_C_IID_Sp-DFA_seed{s}' for s in range(91001,91008)]
# Only AST integer constants in the metadata constructor are rebound. No bare digit
# substitution; hashes, utf-8, resource threads and scientific body are not edited.
numbers={136:140,140:147,40:47,660:653,62:86,8:4,4:7}
lines=source.splitlines(keepends=True);edits=[]
for n in ast.walk(ast.parse(source)):
 if isinstance(n,ast.Constant) and type(n.value) is int and n.value in numbers:
  edits.append((n.lineno-1,n.col_offset,n.end_col_offset,str(numbers[n.value])))
for line,start,end,value in sorted(edits,reverse=True):lines[line]=lines[line][:start]+value+lines[line][end:]
source=''.join(lines)
changes={
 'celeba_mechanism_valid_C_after28_20261009':'celeba_mechanism_valid_C_after36_20261010',
 'celeba_mechanism_valid_C_after36_20261010':'celeba_mechanism_valid_C_after40_20261010',
 'C_after25':'C_after28','C_after28':'C_after36','C_after36':'C_after40',
 'C_AFTER28':'C_AFTER36','C_AFTER36':'C_AFTER40',
 'inventory_actual136_Full100refs.json':'inventory_actual140_Full100refs.json',
 'inventory_actual140_Full100refs.json':'inventory_actual147_Full100refs.json',
 '1d06eca4b5eccf94fa5c510ca867d4281e0a8fe3bce5a2b17b18864e43bdcc67':'450ae61e37432f6651da1a594ed5b9f701c465282435a3d8242664ae228d4509',
 'c823dce69ef511ff15faf1b964805184a6a8c10f271214de1cc6706446a49ba2':'b1e6467eac5b7218cda6af189c2ae2b655fb80780d48db05b45463aa9bdb578f',
 '7cf324b92be73420b7f4497673519c779666664d13a98ad93f4207434c1de635':'ddda34996da47f176520e782ca03ebf536bda7b1a361b6e34b70092e660299c1',
 'incremental_20261009T221150Z':'incremental_20261009T225335Z',
 '0d1661bdc21025b957fa4e6ac39d4c8924cb50ab1b91920268119941312a615a':'cd197e57b42dda87bc239736e401cb5accd029313546533f6a48caf27950e1e0',
 'cd197e57b42dda87bc239736e401cb5accd029313546533f6a48caf27950e1e0':'eb1f6c3120febb138e32af25484ac790cf962cd15d5ea9921140119bc5a3317a',
 'ACTUAL_STRICT_OFFSERVER_NATIVE140_C_AFTER36_INPUTS':'ACTUAL_STRICT_OFFSERVER_NATIVE147_C_AFTER40_INPUTS',
 'original128_unchanged':'original136_unchanged',
 'original236_raw_json_record_bytes_exact':'original240_raw_json_record_bytes_exact',
 "review['added4']":"review['added_ids']",
 'prior128':'prior136','prior136':'prior140','Prior128':'Prior136','Prior136':'Prior140',
 'closed128':'closed136','closed136':'closed140','PRIOR136':'PRIOR140',
 'original136_records_exact':'original140_records_exact',
 'selected8_pairs':'selected4_pairs','selected4_pairs':'selected7_pairs',
 'SELECTED_4.txt':'SELECTED_7.txt','exact4':'exact7','selected4':'selected7',
 'selected8':'selected4','prepared8 frozen':'prepared4 frozen','prepared4 frozen':'prepared7 frozen',
 'reviewed8 root':'reviewed4 root','reviewed4 root':'reviewed7 root',
 'Only8 IDs':'Only4 IDs','Only4 IDs':'Only7 IDs',
 'reviewed8 CPU':'reviewed4 CPU','reviewed4 CPU':'reviewed7 CPU',
 'Prepared exact8-terminal':'Prepared exact4-terminal','Prepared exact4-terminal':'Prepared exact7-terminal',
 'exact8-scope':'exact4-scope',
 'len(records) == 136':'len(records) == 140','len(records) == 140':'len(records) == 147',
 "for r in records}) == 136":"for r in records}) == 140","for r in records}) == 140":"for r in records}) == 147",
 'exactly136 actual accepted terminals':'exactly140 actual accepted terminals','exactly140 actual accepted terminals':'exactly147 actual accepted terminals',
 'Actual accepted136 snapshot':'Actual accepted140 snapshot','Actual accepted140 snapshot':'Actual accepted147 snapshot',
 'Excluded-prior128/selected8':'Excluded-prior136/selected4','Excluded-prior136/selected4':'Excluded-prior140/selected7',
 '== 664':'== 660','== 660':'== 653',
 'Only adopted U100+C28 plus exact eight C terminals; no other scope':'Only adopted U100+C36 plus exact four C terminals; no other scope',
 'Only adopted U100+C36 plus exact four C terminals; no other scope':'Only adopted U100+C40 plus exact seven C terminals; no other scope',
 'len(ids) == len(set(ids)) == 8':'len(ids) == len(set(ids)) == 4',
 'len(ids) == len(set(ids)) == 4':'len(ids) == len(set(ids)) == 7',
 "len(scope['excluded_prior_ids']) == 128":"len(scope['excluded_prior_ids']) == 136",
 "len(scope['excluded_prior_ids']) == 136":"len(scope['excluded_prior_ids']) == 140",
 'len(chosen)==8':'len(chosen)==4','len(chosen)==4':'len(chosen)==7',
 'len(prior|set(ids))<8':'len(prior|set(ids))<4','len(prior|set(ids))<4':'len(prior|set(ids))<7',
 'ALL8_STRICT':'ALL4_STRICT','ALL4_STRICT':'ALL7_STRICT',
 'native140 minus closed136 C4':'native147 minus closed140 C7',
 'native C minus closed U100+C36':'native C minus closed U100+C40',
 '660 not yet native accepted':'653 not yet native accepted',
 "prior128_root_adoption_sha256":"prior136_root_adoption_sha256",
}
source=re.sub('|'.join(re.escape(k) for k in sorted(changes,key=len,reverse=True)),lambda m:changes[m.group(0)],source)
node=next(n for n in ast.parse(source).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SELECTED' for t in n.targets))
source=source.replace(ast.get_source_segment(source,node),'SELECTED = '+repr(selected),1)
# New boundary retains the actual parent's prior136 lineage field, not a fictitious
# prior140 field inside the historical parent adoption receipt.
source=source.replace("'prior140_root_adoption_sha256':adoption['prior140_root_adoption_sha256']", "'prior136_root_adoption_sha256':adoption['prior136_root_adoption_sha256']")
source=source.replace("s=s.replace('Previously accepted C after25','Previously accepted C after28')", "s=s.replace('Previously accepted C after28','Previously accepted C after36')")
ast.parse(source)
assert 'utf-8' in source and 'utf-4' not in source
with (H/'prepare.py').open('x',encoding='utf-8',newline='\n') as f:f.write(source)
B='docs/server_deployment_20260923/training_20260923/server_reactivation_20261009/mechanism_science_backups_20261009/'
tag='root_delta_20261009T230901Z'
files={
 'archive':(B+tag+'.tar.gz','da22ea858d18c047eb2e9ea108a46bafab19328b7672aed51dc921e703d03d77'),
 'receipt':(B+tag+'.tar.gz.receipt.json','adcb90be6676aa7193550b2578d941401207f144dea6e3776d28dbb95e2d3fb0'),
 'proof':(B+tag+'_offserver_verification.json','33f88249d57c7032872d9aeaea8e2b265ad252a51617be5a3677339d9bcffecd'),
 'ledger':(B+tag+'/verified_ledger.json','dc92864bf8d536c0694580cce4676472a0e2388ec99e6eb6fc5957ac501229fd'),
 'inspection':(B+'mechanism_inspection_v4_'+tag+'/inspection.json','bcda76141f61f88c13c177f3cdee9853310001d8dedc74bad3bf9b3f7af3f285'),
 'root_delta':(B+tag+'/ROOT_DELTA_VERIFICATION.json','b8b5addc893455142ef1ebefd0e2dac8203bc7c18ad7ce09bff966aa8d20116d'),
 'root_review':((H/'ROOT_NATIVE_INCREMENT_REVIEW.json').relative_to(R).as_posix(),'5c01316cac5008fa3ec5fa21203fee8f0b7f0f94bd5bb0793bd9f4437a7627d5')}
ni={'status':'ACTUAL_STRICT_OFFSERVER_NATIVE147_C_AFTER40_INPUTS','selected_ids':selected,'files':{n:{'path':p,'sha256':s} for n,(p,s) in files.items()},'actual_native_accepted':147,'prior_three_views_accepted':140,'CNN':False,'dispatch':False}
with (H/'NATIVE_INPUTS.json').open('x',encoding='utf-8',newline='\n') as f:json.dump(ni,f,indent=2);f.write('\n')
print(json.dumps({'prepare_sha256':hashlib.sha256(source.encode()).hexdigest(),'selected':len(selected),'native_accepted':147,'prior_accepted':140,'CNN':False}))
