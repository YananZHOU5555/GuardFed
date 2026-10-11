"""Original zero-fit consumer; only actual output namespace changes."""
from pathlib import Path
import hashlib,importlib.util
HERE=Path(__file__).resolve().parent;C=HERE.parent
p=C/'saved_v2/audit_saved.py'
assert hashlib.sha256(p.read_bytes()).hexdigest()=='fe437f62b3a781261f906d811c9b7fd695a05fd701c86d9af9a4712896005183'
spec=importlib.util.spec_from_file_location('_FedSA10_original_zero_fit',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
m.ACTUAL=C/'actual/saved002'
m.OUTPUT=m.ACTUAL/'SAVED_OUTPUTS_AUDIT_NO_REFIT.json'
m.STARTED=m.ACTUAL/'SAVED_OUTPUTS_AUDIT_NO_REFIT.started.json'
m.FAILED=m.ACTUAL/'SAVED_OUTPUTS_AUDIT_NO_REFIT.failure.json'
if __name__=='__main__':m.main()
