"""Explicit source-only rebind of accepted C-after40 to the fixed actual three IDs."""
from pathlib import Path
import ast,hashlib,json,re
H=Path(__file__).resolve().parent;R=H.parents[1];O=H.with_name('celeba_mechanism_valid_C_after40_20261010')
source=(O/'prepare.py').read_text(encoding='utf-8-sig')
selected=[f'minus_C_IID_Sp-DFA_seed{s}' for s in range(91008,91011)]
numbers={140:147,147:150,47:50,653:650,86:54,7:3,4:7}
lines=source.splitlines(keepends=True);edits=[]
for n in ast.walk(ast.parse(source)):
 if isinstance(n,ast.Constant) and type(n.value) is int and n.value in numbers:
  edits.append((n.lineno-1,n.col_offset,n.end_col_offset,str(numbers[n.value])))
for line,start,end,value in sorted(edits,reverse=True):lines[line]=lines[line][:start]+value+lines[line][end:]
source=''.join(lines)
changes={
 'celeba_mechanism_valid_C_after28_20261009':'celeba_mechanism_valid_C_after36_20261010',
 'celeba_mechanism_valid_C_after36_20261010':'celeba_mechanism_valid_C_after40_20261010',
 'celeba_mechanism_valid_C_after40_20261010':'celeba_mechanism_valid_C_after47_20261010',
 'C_after28':'C_after36','C_after36':'C_after40','C_after40':'C_after47',
 'C_AFTER36':'C_AFTER40','C_AFTER40':'C_AFTER47',
 'inventory_actual140_Full100refs.json':'inventory_actual147_Full100refs.json',
 'inventory_actual147_Full100refs.json':'inventory_actual150_Full100refs.json',
 '450ae61e37432f6651da1a594ed5b9f701c465282435a3d8242664ae228d4509':'22da16b734f2c6041d494e30c200d586ff79626ac0a5ec6b438bea405c7dc48b',
 'b1e6467eac5b7218cda6af189c2ae2b655fb80780d48db05b45463aa9bdb578f':'1ac84b16f00c30ed3effd230f539824bb426540e66ec4981a53fdee676c50ec8',
 'ddda34996da47f176520e782ca03ebf536bda7b1a361b6e34b70092e660299c1':'c65a6e8a88ef1e97a687bbf259840582c3d1f4c26e8e09deb604e94a4cc5b10f',
 'incremental_20261009T225335Z':'incremental_20261009T233028Z',
 'cd197e57b42dda87bc239736e401cb5accd029313546533f6a48caf27950e1e0':'eb1f6c3120febb138e32af25484ac790cf962cd15d5ea9921140119bc5a3317a',
 'eb1f6c3120febb138e32af25484ac790cf962cd15d5ea9921140119bc5a3317a':'64732337f35f81bed49e40c60fa1d5229fb6a7557c45272505fee488c5ad20ea',
 'ACTUAL_STRICT_OFFSERVER_NATIVE147_C_AFTER40_INPUTS':'ACTUAL_STRICT_OFFSERVER_NATIVE150_C_AFTER47_INPUTS',
 'original136_unchanged':'original140_unchanged',
 'original240_raw_json_record_bytes_exact':'original247_raw_json_record_bytes_exact',
 'prior136':'prior140','prior140':'prior147','Prior136':'Prior140','Prior140':'Prior147',
 'closed136':'closed140','closed140':'closed147','PRIOR140':'PRIOR147',
 'original140_records_exact':'original147_records_exact',
 'selected4_pairs':'selected7_pairs','selected7_pairs':'selected3_pairs',
 'SELECTED_7.txt':'SELECTED_3.txt','exact7':'exact3','selected7':'selected3',
 'selected4':'selected7','prepared4 frozen':'prepared7 frozen','prepared7 frozen':'prepared3 frozen',
 'reviewed4 root':'reviewed7 root','reviewed7 root':'reviewed3 root',
 'Only4 IDs':'Only7 IDs','Only7 IDs':'Only3 IDs',
 'reviewed4 CPU':'reviewed7 CPU','reviewed7 CPU':'reviewed3 CPU',
 'Prepared exact4-terminal':'Prepared exact7-terminal','Prepared exact7-terminal':'Prepared exact3-terminal',
 'exact4-scope':'exact7-scope',
 'len(records) == 140':'len(records) == 147','len(records) == 147':'len(records) == 150',
 "for r in records}) == 140":"for r in records}) == 147","for r in records}) == 147":"for r in records}) == 150",
 'exactly140 actual accepted terminals':'exactly147 actual accepted terminals','exactly147 actual accepted terminals':'exactly150 actual accepted terminals',
 'Actual accepted140 snapshot':'Actual accepted147 snapshot','Actual accepted147 snapshot':'Actual accepted150 snapshot',
 'Excluded-prior136/selected4':'Excluded-prior140/selected7','Excluded-prior140/selected7':'Excluded-prior147/selected3',
 '== 660':'== 653','== 653':'== 650',
 'Only adopted U100+C36 plus exact four C terminals; no other scope':'Only adopted U100+C40 plus exact seven C terminals; no other scope',
 'Only adopted U100+C40 plus exact seven C terminals; no other scope':'Only adopted U100+C47 plus exact three C terminals; no other scope',
 'len(ids) == len(set(ids)) == 4':'len(ids) == len(set(ids)) == 7',
 'len(ids) == len(set(ids)) == 7':'len(ids) == len(set(ids)) == 3',
 "len(scope['excluded_prior_ids']) == 136":"len(scope['excluded_prior_ids']) == 140",
 "len(scope['excluded_prior_ids']) == 140":"len(scope['excluded_prior_ids']) == 147",
 'len(chosen)==4':'len(chosen)==7','len(chosen)==7':'len(chosen)==3',
 'len(prior|set(ids))<4':'len(prior|set(ids))<7','len(prior|set(ids))<7':'len(prior|set(ids))<3',
 'ALL4_STRICT':'ALL7_STRICT','ALL7_STRICT':'ALL3_STRICT',
 'native147 minus closed140 C7':'native150 minus closed147 C3',
 'U100+C36':'U100+C40','U100+C40':'U100+C47',
 '653 not yet native accepted':'650 not yet native accepted',
 'Previously accepted C after28':'Previously accepted C after36','Previously accepted C after36':'Previously accepted C after40',
}
source=re.sub('|'.join(re.escape(k) for k in sorted(changes,key=len,reverse=True)),lambda m:changes[m.group(0)],source)
node=next(n for n in ast.parse(source).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SELECTED' for t in n.targets))
source=source.replace(ast.get_source_segment(source,node),'SELECTED = '+repr(selected),1)
ast.parse(source);assert 'utf-8' in source and 'utf-4' not in source
with (H/'prepare.py').open('x',encoding='utf-8',newline='\n') as f:f.write(source)
B='docs/server_deployment_20260923/training_20260923/server_reactivation_20261009/mechanism_science_backups_20261009/';tag='root_delta_20261009T233607Z'
files={
 'archive':(B+tag+'.tar.gz','11ae8eb7f7cae59e10cd5df79f9681b13ebbdec2c4f460968d93e9fb82e98866'),
 'receipt':(B+tag+'.tar.gz.receipt.json','6af158d14a6f13844d4509c28c3e66c1b5583a1fc239aef7ec7f6af0f6f13951'),
 'proof':(B+tag+'_offserver_verification.json','1a552aa963a6c8497d71165e69a19b74672f7a469adb219c491dc39e070c5a7e'),
 'ledger':(B+tag+'/verified_ledger.json','7e04be828e3b6824dbb13e40b0cc8d3a9435f48541a0fafebc2748f259691353'),
 'inspection':(B+'mechanism_inspection_v4_'+tag+'/inspection.json','5ad6f7753271c882b20a8ecc975a3fbdc9228e8aaa894222e63f80f118adb439'),
 'root_delta':(B+tag+'/ROOT_DELTA_VERIFICATION.json','744c17ea5ef63184fdbbc66d7a95f8277dc3497fd82f6ba76aaa19e6bbb34b65'),
 'root_review':((H/'ROOT_NATIVE_INCREMENT_REVIEW.json').relative_to(R).as_posix(),'c0e78899ce232b81c7c2849ed9d05283c3c740bc962176074ad7281c72c24f4a')}
ni={'status':'ACTUAL_STRICT_OFFSERVER_NATIVE150_C_AFTER47_INPUTS','selected_ids':selected,'files':{n:{'path':p,'sha256':s} for n,(p,s) in files.items()},'actual_native_accepted':150,'prior_three_views_accepted':147,'CNN':False,'dispatch':False}
with (H/'NATIVE_INPUTS.json').open('x',encoding='utf-8',newline='\n') as f:json.dump(ni,f,indent=2);f.write('\n')
print(json.dumps({'prepare_sha256':hashlib.sha256(source.encode()).hexdigest(),'selected':len(selected),'native_accepted':150,'prior_accepted':147,'CNN':False}))
