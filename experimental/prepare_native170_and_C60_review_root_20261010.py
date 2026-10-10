"""Reuse the prior bounded root checks with the new actual identity boundary."""
from pathlib import Path
R=Path(__file__).resolve().parents[1]
def changed(source,target,pairs):
    s=(R/source).read_text('utf8')
    for old,new in pairs:
        assert old in s,old
        s=s.replace(old,new)
    with (R/target).open('x',encoding='utf8') as f:f.write(s)
changed('tmp/adopt_native160_review_root_20261010.py','tmp/adopt_native170_review_root_20261010.py',[
 ('four non-IID Benign','ten non-IID F Flip'),('root_delta_20261010T003657Z','root_delta_20261010T013518Z'),
 ('valid_C_after56_20261010','valid_C_after60_20261010'),
 ('369c5d80169f5188604f46565116a184de3b1c84eaf53ac4bc3794279a0e6ee4','e45971c7f3db346f54594a7eccfc2f41c4bf2a4a7d015fab020a06dc7a52343e'),
 ('minus_C_non-IID_Benign_seed','minus_C_non-IID_F Flip_seed'),('range(91007,91011)','range(91001,91011)'),
 ('OLD256','OLD260'),('==160','==170'),("['added_n']==4","['added_n']==10"),
 ("['archive_members_verified']==62","['archive_members_verified']==110"),
 ("['ledger_entries_verified']==25","['ledger_entries_verified']==26"),
 ('ledger_previous24_entries_exact','ledger_previous25_entries_exact'),('original256','original260'),
 ('==(100,260,60)','==(100,270,70)'),('ROOT_NATIVE160','ROOT_NATIVE170')])
changed('tmp/review_C_after56_source_root_20261010.py','tmp/review_C_after60_source_root_20261010.py',[
 ('exact-four','exact-ten'),('valid_C_after56_20261010','valid_C_after60_20261010'),
 ('valid_C_after50_20261010','valid_C_after56_20261010'),('C_after56_source_review','C_after60_source_review'),
 ('minus_C_non-IID_Benign_seed','minus_C_non-IID_F Flip_seed'),('range(91007,91011)','range(91001,91011)'),
 ('6fd1881566baf4b7de107752f07354bf8f591f117681e385a12fa1e92307235a','5d35280b387c7ff621d94c00fb5983284edf06f272dba4f5a01dc78fb297f824'),
 ('2095ab384a7844355fc92453dbfa5d2922f88f378838e7b7eda2524578f3bbd6','ed8ecc84781b799e205c9e139dce8e753cb5545139b7e48ef23d5f8d07a50a77'),
 ('d1679c0bbd53bc66e4ea7ae792000d398efcafe5192bc5164ffd80e7a2eeb236','12c1b612d9c9697f6a537344ebace5aaa7c1bf5d7bbe760a3866168adca89896'),
 ('302dd45e9f05c646671d31d26775607af7a4fe70876fa1e60643939f972742f4','c28047e56778d6285140dcb22386410a0fc3dd3a803a9bd9aba182be373c1cef'),
 ('inventory_actual160_Full100refs.json','inventory_actual170_Full100refs.json'),
 ('inventory_actual156_Full100refs.json','inventory_actual160_Full100refs.json'),
 ('len(before)==156','len(before)==160'),('len(current[\'records\'])==160','len(current[\'records\'])==170'),
 ('per_child_exact4_science_approval_positive\']==4','per_child_exact10_science_approval_positive\']==10'),
 ('369c5d80169f5188604f46565116a184de3b1c84eaf53ac4bc3794279a0e6ee4','e45971c7f3db346f54594a7eccfc2f41c4bf2a4a7d015fab020a06dc7a52343e'),
 ("['native_accepted']==160","['native_accepted']==170"),
 ('EXACT4_APPROVAL','EXACT10_APPROVAL'),('native_accepted_snapshot=160','native_accepted_snapshot=170'),
 ('excluded_prior_three_view_ids=156','excluded_prior_three_view_ids=160'),('old156','old160'),
 ('positive_approval_exact4','positive_approval_exact10'),('original_C_after56','original_C_after60'),
 ('prior156_root_adoption','prior160_root_adoption'),
 ('a7231a724ea3a4d3443bbe196137b0c72db025cbc98d3ac66842334ab0fe9080','21b7f5beac762bf808415685817a4027e9da66640a7ef0e4c7d4ac74eeb1f05e')])
print('Two bounded root check scripts prepared; nothing executed')
