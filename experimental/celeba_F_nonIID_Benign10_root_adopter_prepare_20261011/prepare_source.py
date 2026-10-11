"""Source-only derivation of the existing exact10 root adopter; no adoption or scientific execution."""
from pathlib import Path
import ast,difflib,hashlib,json
H=Path(__file__).resolve().parent;R=H.parents[1]
P=R/'tmp/celeba_remaining_F_SpDFA10_prepared_20261011/adopt_by_root.py'
S=R/'tmp/celeba_mechanism_remaining_F_nonIID_Benign10_prepare_20261011'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())
def put(n,x):
    raw=x if isinstance(x,bytes) else (x if isinstance(x,str) else json.dumps(x,ensure_ascii=False,indent=2)+'\n').encode('utf8')
    with (H/n).open('xb') as f:f.write(raw)
assert sha(P)=='1aae42dfbd0376148b7fc5331a856baa6175b04b829cb257b63908ee1d9d9a74'
assert sha(S/'MECHANISM_INDEX.json')=='16299ca4ade394a1b89edb3f9230845cda7ce826c783645b9aada4ea359a5c37'
assert sha(S/'NATIVE_IDENTITY_JOIN.json')=='8050233036aaf1ee46085f1930fbd9ec76e8f05ff7cef0f2c2937e25838e6949'
e=read(S/'EXECUTION_INPUTS.json')
assert e['root_native_total']==360 and e['native_root_sha256']=='396e40654b6dfc75029d4a7a42dbfafe4caaf111a872ae2ece5045f6d1203418'
old=P.read_bytes();s=old.decode('utf8');changes=[]
def change(a,b,role='metadata'):
    global s
    assert s.count(a)==1,(a,s.count(a))
    s=s.replace(a,b,1);changes.append(dict(role=role,old=a,new=b))
change('retaining the prior340 evidence chain','retaining the prior350 evidence chain')
change("S=R/'tmp/celeba_remaining_F_SpDFA10_prepared_20261011'", "S=R/'tmp/celeba_mechanism_remaining_F_nonIID_Benign10_prepare_20261011'\nsys.path.insert(0,str(S))",'namespace_and_import_path')
change("D=R/'tmp/celeba_mechanism_remaining_F_SpDFA10_root_adoption_20261011'", "D=R/'tmp/celeba_mechanism_remaining_F_nonIID_Benign10_root_adoption_20261011'")
change("assert actual['native_total']==350 and actual['replay_total']==350 and actual['new_replays']==10", "assert actual['native_total']==360 and actual['replay_total']==360 and actual['new_replays']==10\nassert actual['native_root_sha256']=='396e40654b6dfc75029d4a7a42dbfafe4caaf111a872ae2ece5045f6d1203418'")
change("expected=[f'minus_F_IID_Sp-DFA_seed{s}' for s in range(91001,91011)]", "expected=[f'minus_F_non-IID_Benign_seed{s}' for s in range(91001,91011)]")
change('==(340,10,350,160,170)','==(350,10,360,170,180)')
change('a92ca53bf7beb7fac0951381ba1807ff0ee4f6d74d040826b0137cfb700d1d99','5b24956b4f14707e2a284fe73ffb538545b4aad218c833e31427720402ff02b3')
change('80a7f38890c2a7f8c75c0472489315bb2acb9e43c9e28d686a799cfea9709a02','3f7b436e0b2fec0401d7bf6117d6dca48aef21c185b6f170d37954e2cafee5a4')
change("read(prior_root)['cumulative_accepted']==340","read(prior_root)['cumulative_accepted']==350")
change("len(set(index['all_ids']))==350","len(set(index['all_ids']))==360")
change("join['prior340_objects_and_order_unchanged'] and join['old441_native_records_exact']", "join['prior350_objects_and_order_unchanged'] and join['old450_native_records_exact']")
change("len(receipt['all_transported_ids'])==170","len(receipt['all_transported_ids'])==180")
change('4517867eeb061590c9e5aba8e6f63315c5e28b7a97bb9e9a53e38a7b01dd4524','d40f806a27db0512647edc2206b8881e5d407a4a655c8dde1a47b7b001ee75b9')
change("['native341_root_path']","['native350_root_path']")
change("len(old_inspection['records'])==441","len(old_inspection['records'])==450")
change("assert len(ledger['entries'])==49 and ledger['entries'][:-1]==old_ledger['entries']", "assert len(old_ledger['entries'])==49 and len(ledger['entries'])==50 and ledger['entries'][:-1]==old_ledger['entries']")
identity_old="==('minus_F','IID',row['attack'],5000,70,'valid',19867) and row['attack']=='Sp-DFA'"
identity_new="==('minus_F','non-IID',row['attack'],5,70,'valid',19867) and row['attack']=='Benign'"
change(identity_old,identity_new,'scene_identity_only_distribution_attack_true_alpha')
line="assert {int(i[-5:]) for i in index['all_ids'] if i.startswith('minus_F_IID_Sp-DFA_seed')}==set(range(91001,91011))"
change(line,line+"\nassert {int(i[-5:]) for i in index['all_ids'] if i.startswith('minus_F_non-IID_Benign_seed')}==set(range(91001,91011))")
change("target=D/'MECHANISM350_INDEX.json'","target=D/'MECHANISM360_INDEX.json'")
change("status='ROOT_F_SPDFA10_EXACT10_SAVED_ARRAYS_REPLAY350_ADOPTED'","status='ROOT_F_NONIID_BENIGN10_EXACT10_SAVED_ARRAYS_REPLAY360_ADOPTED'")
change('prior_accepted=340,cumulative_accepted=350,remaining620_new_accepted=170','prior_accepted=350,cumulative_accepted=360,remaining620_new_accepted=180')
change('original340_unchanged=True','original350_unchanged=True')
change("complete_F_scenes=[['IID','Benign'],['IID','F Flip'],['IID','FedSA'],['IID','S-DFA'],['IID','Sp-DFA']]", "complete_F_scenes=[['IID','Benign'],['IID','F Flip'],['IID','FedSA'],['IID','S-DFA'],['IID','Sp-DFA'],['non-IID','Benign']]")
change("parent_adopter_path='tmp/celeba_remaining_F_SDFA10_prepared_20261011/adopt_by_root.py',parent_adopter_sha256=sha(R/'tmp/celeba_remaining_F_SDFA10_prepared_20261011/adopt_by_root.py')", "parent_adopter_path='tmp/celeba_remaining_F_SpDFA10_prepared_20261011/adopt_by_root.py',parent_adopter_sha256=sha(R/'tmp/celeba_remaining_F_SpDFA10_prepared_20261011/adopt_by_root.py')")
change('accepted=350,A=100','accepted=360,A=100')
new=s.encode('utf8');compile(new,str(H/'adopt_root.py'),'exec');compile(Path(__file__).read_bytes(),__file__,'exec')
inverse=s
for item in reversed(changes):
    assert inverse.count(item['new'])==1,item['role']
    inverse=inverse.replace(item['new'],item['old'],1)
assert inverse.encode('utf8')==old
assert ast.dump(ast.parse(inverse),include_attributes=False)==ast.dump(ast.parse(old),include_attributes=False)
def block(raw,a,b):return raw[raw.index(a):raw.index(b,raw.index(a))]
record_old=block(old,b'for rec in records:',b'release=read(')
record_new=block(new,b'for rec in records:',b'release=read(')
assert record_old==record_new
archive_old=block(old,b'for na in index[',b'assert len(native_checks)==20')
archive_new=block(new,b'for na in index[',b'assert len(native_checks)==20')
assert archive_new.replace(identity_new.encode(),identity_old.encode())==archive_old
assert ast.dump(ast.parse(record_new),include_attributes=False)==ast.dump(ast.parse(record_old),include_attributes=False)
assert ast.dump(ast.parse(archive_new.replace(identity_new.encode(),identity_old.encode())),include_attributes=False)==ast.dump(ast.parse(archive_old),include_attributes=False)
member_old=block(old,b'for name,pin in files.items():',b'hp=S/')
member_new=block(new,b'for name,pin in files.items():',b'hp=S/')
assert member_old==member_new
verify_old=block(old,b"original=R/'tmp/celeba_mechanism_evidence",b'native_root=Path(')
verify_new=block(new,b"original=R/'tmp/celeba_mechanism_evidence",b'native_root=Path(')
assert verify_old==verify_new
put('adopt_root.py',new)
put('SOURCE_DIFF.patch',''.join(difflib.unified_diff(old.decode('utf8').splitlines(True),s.splitlines(True),fromfile='parent_SpDFA10/adopt_by_root.py',tofile='nonIID_Benign10/adopt_root.py')))
put('SOURCE_ADAPTATIONS.json',changes)
check=dict(status='SOURCE_ONLY_ORIGINAL_ADOPTER_METADATA_ADAPTATION_COMPILE_RAW_AST_PASS_NOT_EXECUTED',parent_source_path=str(P.relative_to(R)),parent_source_sha256=sha(P),candidate_source_sha256=sha(H/'adopt_root.py'),metadata_spans=len(changes),complete_inverse_to_parent_raw_bytes_exact=True,complete_inverse_AST_exact=True,original_per_record_receipt_identity_tolerance_loop_raw_exact=True,original_archive_member_loop_raw_exact_after_one_scene_identity_inverse=True,original_delivery_member_loop_raw_exact=True,original_archive_verifier_block_raw_exact=True,scene_identity_allowed_change=dict(distribution=['IID','non-IID'],attack=['Sp-DFA','Benign'],actual_alpha=[5000,5],rounds=70,split='valid',n_eval=19867),import_path_change='Add exact S to sys.path because adopter moved outside saved-source directory; original verifier source/function unchanged',prior350_index_sha256='5b24956b4f14707e2a284fe73ffb538545b4aad218c833e31427720402ff02b3',actual_native360_sha256=e['native_root_sha256'],actual_proposed_index_sha256=sha(S/'MECHANISM_INDEX.json'),actual_join_sha256=sha(S/'NATIVE_IDENTITY_JOIN.json'),future_actual_delivery_sha256=None,future_root_actual_pins_sha256=None,archive_members_dynamic_not_guessed=True,original_1e12_guard_unchanged=True,compiled=['adopt_root.py','prepare_source.py'],SSH=0,model_array_reads=0,F_write=0,adoption_executed=False,STATE_written=False,Git_mutations=0)
put('SOURCE_CHECK.json',check)
put('ROOT_ACTUAL_PINS.template.json',dict(status='TEMPLATE_ONLY_ROOT_MUST_BIND_ACTUAL_FINALIZED_DELIVERY',native_root_path=e['native_root_path'],native_root_sha256=e['native_root_sha256'],native_total=360,replay_total=360,new_replays=10,authorized_snapshot_path=e['authorized_snapshot_path'],authorized_snapshot_sha256=e['authorized_snapshot_sha256'],delivery_sha256=None,delivery_members=None,handoff_sha256=None,proposed_index_sha256=sha(S/'MECHANISM_INDEX.json'),archive_sha256=None,archive_members=None,offserver_sha256=None,source_adopter_sha256=None,root_actual_pins_sha256=None,execution_authorized=False))
print(json.dumps(dict(status=check['status'],source_sha256=check['candidate_source_sha256'],spans=len(changes),inverse_raw_exact=True)))
