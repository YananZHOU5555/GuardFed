"""Rebind strict completed-transfer helpers to the actual exact8 startup."""
from pathlib import Path
import ast, hashlib, json

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'tmp/celeba_mechanism_valid_incremental_after92_20261009'
EX = BASE / 'execution_candidate'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_bytes())
assert sha(EX/'ROOT_STARTUP_OBSERVATION.json') == 'c773018542610620d1640f21bf8b8c443b7bff442898a922fc56ea93ee228030'
assert sha(EX/'deployment_receipt.json') == 'd0f7d912530819fc5e230cf4f0478e77fd16e8dc957688b6ace56af89d7be4fa'
root_approval = read(EX/'deployment_receipt.json')['root_approval_sha256']
assert root_approval == sha(EX/'ROOT_APPROVED.json')
pins = [
    ('b95801ac039bb39276e79012393792adfde290b92091ff45c0a4323bd4fdd8f0', sha(BASE/'FILES_SHA256.json')),
    ('94d99842b334346ae8a6715b84f7fddea35574a77fbf51f9460b25301c6ccae6', sha(EX/'EXECUTION_SOURCE_SHA256.json')),
    ('bedb867ae9bdf965ef7143a5b4f077620911ec26d93346337307046e64b5c9da', sha(BASE/'inventory_actual100_Full100refs.json')),
    ('a7fe2e43cadd5f148055720cb9fc00ea5a82e5f133611225187bc07419f0229b', sha(EX/'ROOT_STARTUP_OBSERVATION.json')),
    ('34e5d9ec34a1ddad4bda4e49f746ffeb59545e064311b4b05e8d6d082731e5a0', root_approval),
]
results = []
for role in ('backup', 'adopt'):
    old = ROOT/f'tmp/{role}_mechanism_after82_v2_root_20261009.py'
    code = old.read_text(encoding='utf8')
    code = code.replace('after82_v2', 'after92').replace('AFTER82_V2', 'AFTER92').replace('exact10', 'exact8')
    code = code.replace('actual92_Full100refs', 'actual100_Full100refs').replace('==10', '==8')
    for before, after in pins:
        assert before in code, (role, before)
        code = code.replace(before, after)
    if role == 'adopt':
        for before, after in [
            ('tmp/celeba_mechanism_valid_incremental_after71_20261009/execution_candidate/backups/incremental_20261009T163050Z/ROOT_ADOPTION_REVIEW.json', 'tmp/celeba_mechanism_valid_incremental_after82_v2_20261009/execution_candidate/backups/incremental_20261009T174922Z/ROOT_ADOPTION_REVIEW.json'),
            ('fb2bc745e7a6b6aa3d7d4cb898aefa31c6acbf73bee7a8e1a30ab667e03c204c', 'b9e40d1ca565c0bcf146058433ff3e037ab4e824aa6972d1a3f3f47a088e8683'),
            ("read(prior)['cumulative_three_view_models']==82", "read(prior)['cumulative_three_view_models']==92"),
            ("len(set(scope['excluded_prior_ids']))==82", "len(set(scope['excluded_prior_ids']))==92"),
            ('(90,240,30)', '(72,192,24)'),
            ('prior_three_view_models=82,accepted_new=10,cumulative_three_view_models=92', 'prior_three_view_models=92,accepted_new=8,cumulative_three_view_models=100'),
            ('original82_unchanged', 'original92_unchanged'),
            ('prior82_root_adoption_sha256', 'prior92_root_adoption_sha256'),
            ('and prior71.', 'and prior92.'),
        ]:
            assert before in code, (role, before)
            code = code.replace(before, after)
    ast.parse(code)
    target = ROOT/f'tmp/{role}_mechanism_after92_root_20261009.py'
    with target.open('x', encoding='utf8', newline='\n') as stream:
        stream.write(code)
    results.append(dict(role=role, source_sha256=sha(old), path=target.relative_to(ROOT).as_posix(), sha256=sha(target)))
print(json.dumps(dict(status='ROOT_AFTER92_STRICT_TRANSFER_HELPERS_PREPARED_NOT_EXECUTED', helpers=results)))
