"""Bind the original FL96 adopter to the sealed two-job increment after sixteen."""
from pathlib import Path
import ast, hashlib, re

ROOT = Path(__file__).resolve().parents[1]
original = ROOT / 'tmp/adopt_FL96_after14_delta_root_20261010.py'
assert hashlib.sha256(original.read_bytes()).hexdigest() == '102d484268adb968ca33b127c7647ae9af64e2d3fc8ecef953b95fccc26484a9'
old = original.read_text(encoding='utf8')
bindings = {
    'celeba_flgmm_fullcoverage_delta_after14_20261010': 'celeba_flgmm_fullcoverage_delta_after16_20261010',
    '0b07017f3ab52fdec86e7924b6f8e45a3055c25e774f0230448a5c78dc36bd6d': 'a1f32d08137d47420da93e9e8b35626fe17686644f472b7ccfda0fc8d78e35bd',
    '5d3a886304504c898a4f70067302599042db72c0fb163dfe18c3f0bb8c1b48d6': '78da36a9fcc6e278152f364a8251aa46e47c5b44614538ddbfe1d90c2b7cc575',
    '68a6f8d9aa7d4716094a999cfe0938b6d45ec7b4a4223f3f6a85bfcff451737e': 'f21c799c9bb15c784e240ce30ff4699207c0c329e9a5ec4a3872162ff9ff25d7',
    '801ec992899529dc7b38e68acfde3b101d2b90a7966def64f0bbf901cf88793b': '9d587b8261c2d1340339e1905db133bc113aeff7b7150db95cbe260103c77617',
    '2686c5c4d1121611d888462f1e288720465237ae0b80b069fc9e03a38a50513c': '8ca5e7afa10527fd01607082b0a461f0a226a4941091f182ba8fd966cd09197d',
    '8ca5e7afa10527fd01607082b0a461f0a226a4941091f182ba8fd966cd09197d': '48981b380f8b9476f0e168d47b47c4e488feaf69edf8498904ae6084f836383e',
    '(91006, 91007)': '(91008, 91009)',
    "latest['accepted_total'] == 14": "latest['accepted_total'] == 16",
    "handoff['accepted_new_cumulative'] == 16": "handoff['accepted_new_cumulative'] == 18",
    'accepted_before=14, accepted_new=2, accepted_total=16': 'accepted_before=16, accepted_new=2, accepted_total=18',
    'next_latest = dict(accepted_total=16,': 'next_latest = dict(accepted_total=18,',
    # The new handoff omits this redundant field; keep the original pinned
    # helper seal and every helper member check immediately following it.
    " == handoff['original_helper_seal_sha256']": '',
}
for before in bindings:
    assert old.count(before) == 1, before
pattern = '|'.join(re.escape(k) for k in sorted(bindings, key=len, reverse=True))
bound = re.sub(pattern, lambda m: bindings[m.group()], old)
ast.parse(bound)
assert not (ROOT / 'tmp/celeba_flgmm_fullcoverage_delta_after16_20261010/ROOT_ADOPTION_REVIEW.json').exists()
exec(compile(bound, str(original), 'exec'), {'__file__': str(original), '__name__': '__main__'})
