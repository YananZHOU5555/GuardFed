"""Prepare a metadata-only exact12 invocation of the unchanged original transport."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / 'tmp/celeba_remaining620_C19_transport_20261010'
NEW = ROOT / 'tmp/celeba_remaining620_A12_transport_20261010'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_bytes())
prior = read(OLD / 'RAW_STORAGE_INDEX.json')
assert sha(prior['receipt']) == prior['receipt_sha256'] == 'a64d4a97d6201f011a2a3e336984a130745191f71c51f495090b62fca7c5af77'
assert len(prior['all_transported_ids']) == 20
remote_prior = '/workspace/guardfed_checks/celeba_mechanism_remaining_evaluation_transport_20261010/exports/C19_20261010T050336880929Z/backup_receipt.json'
selected = [f'minus_A_IID_Benign_seed{s}' for s in range(91001, 91011)] + [f'minus_A_IID_F Flip_seed{s}' for s in (91001, 91002)]

def change(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)

pre = (OLD / 'remote_preflight.py').read_text(encoding='utf8')
pre = change(pre, 'exact19/CPU110', 'exact12/CPU110')
pre = change(pre,
    "ids=[f'minus_C_non-IID_S-DFA_seed{s}' for s in range(91002,91011)]+[f'minus_C_non-IID_Sp-DFA_seed{s}' for s in range(91001,91011)]",
    'ids=' + repr(selected))
pre = change(pre, 'len(ids)==19', 'len(ids)==12')
pre = change(pre,
    '/workspace/guardfed_checks/celeba_mechanism_remaining_evaluation_transport_20261010/exports/first_20261010T043359656187Z/backup_receipt.json',
    remote_prior)
pre = change(pre, '71e1c6782e175f81b89776d2004bde255d7d09f3bfa99117d5af6e2545bb00fe', prior['receipt_sha256'])
pre = change(pre, "latest['all_transported_ids']==['minus_C_non-IID_S-DFA_seed91001']", "latest['all_transported_ids']==" + repr(prior['all_transported_ids']))
pre = change(pre, 'EXACT19_CLOSED_NO_SELECTED_WORKER_CPU110_AVAILABLE', 'EXACT12_CLOSED_NO_SELECTED_WORKER_CPU110_AVAILABLE')

export = (OLD / 'export_once.py').read_text(encoding='utf8')
export = change(export, 'original-CLI C19 export', 'original-CLI A12 export')
export = change(export, 'EXACT19_CLOSED_NO_SELECTED_WORKER_CPU110_AVAILABLE', 'EXACT12_CLOSED_NO_SELECTED_WORKER_CPU110_AVAILABLE')
export = change(export, "len(pre['selected_ids'])==19", "len(pre['selected_ids'])==12")
export = change(export, "tag='C19_'", "tag='A12_'")
export = change(export,
    "receipt['all_transported_ids']==['minus_C_non-IID_S-DFA_seed91001',*pre['selected_ids']]",
    "receipt['all_transported_ids']==pre['previous_receipt']['all_transported_ids']+pre['selected_ids']")

verify = (OLD / 'download_verify_once.py').read_text(encoding='utf8')
verify = change(verify, "'remaining620_C19_transport_20261010'", "'remaining620_A12_transport_20261010'")
verify = change(verify,
    "first=read(ROOT/'tmp/celeba_remaining620_first_transport_recovery_20261010/HANDOFF.json')",
    "first=read(ROOT/'tmp/celeba_remaining620_C19_transport_20261010/RAW_STORAGE_INDEX.json')")
verify = change(verify, '71e1c6782e175f81b89776d2004bde255d7d09f3bfa99117d5af6e2545bb00fe', prior['receipt_sha256'])
verify = change(verify, "saved['accepted_n']==19", "saved['accepted_n']==12")
verify = change(verify, '==(171,456,57)', '==(108,288,36)')
verify = change(verify, 'F_ONLY_NEW19_ARCHIVE_RECEIPT', 'F_ONLY_NEW12_ARCHIVE_RECEIPT')
assert verify.count('metrics=171,counts=456,rules=57') == 2
verify = verify.replace('metrics=171,counts=456,rules=57', 'metrics=108,counts=288,rules=36')
verify = change(verify, 'new=19,cumulative=20', 'new=12,cumulative=32')

NEW.mkdir(exist_ok=False)
sources = {'remote_preflight.py': pre, 'export_once.py': export, 'download_verify_once.py': verify}
for name, body in sources.items():
    compile(body, str(NEW / name), 'exec')
    with (NEW / name).open('x', encoding='utf8', newline='\n') as stream:
        stream.write(body)
proof = dict(status='PREPARED_ROOT_METADATA_ONLY_EXACT12_TRANSPORT_INVOCATIONS',
    original_source_paths={n: str(OLD / n) for n in sources},
    original_source_sha256={n: sha(OLD / n) for n in sources},
    invocation_sha256={n: sha(NEW / n) for n in sources},
    selected_ids=selected, previous_receipt=prior['receipt'], previous_receipt_sha256=prior['receipt_sha256'],
    prior_transported_ids=prior['all_transported_ids'], original_transport_source_unchanged=True,
    changes='Exact IDs/counts/previous receipt and destination metadata only; original transport/verify science is not modified.',
    new_CNN=0, new_fits=0, new_training=0, test=False, actual_export=False, actual_acceptance=False)
with (NEW / 'PREPARED.json').open('x', encoding='utf8') as stream:
    json.dump(proof, stream, indent=2); stream.write('\n')
print(json.dumps(dict(path=str(NEW), selected_n=len(selected), actual_export=False)))
