"""Namespace-only reuse of the original strict delta collector after adopted341."""
from pathlib import Path
import hashlib
R=Path(__file__).resolve().parents[2]
assert hashlib.sha256(Path(__file__).with_name('PARENT_LEDGER.json').read_bytes()).hexdigest()=='58f108b1cba698559ff96902c9ebe87412061e833421a7dfa0b5e95e99f85898'
P=R/'tmp/celeba_native_after288_20261011/run_once.py'
assert hashlib.sha256(P.read_bytes()).hexdigest()=='9745525453d797b72a36aee66bade4fb3d3db04345fecd93535b68f44d142e88'
text=P.read_text(encoding='utf8')
for old,new in [('native_delta_after288_20261011','native_delta_after341_20261011'),('celeba_native_after288_20261011','celeba_native_after341_20261011')]:
    assert text.count(old)>0
    text=text.replace(old,new)
exec(compile(text,str(P)+':NAMESPACE_ONLY_AFTER341','exec'),dict(__name__='__main__',__file__=__file__))
