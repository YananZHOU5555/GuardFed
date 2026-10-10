"""Fixed metadata guards only. No scientific imports or execution on import."""
from pathlib import Path
import datetime, hashlib, json, re, subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CANDIDATE = ROOT/'tmp/fl_three_view_after48_20261011'
PACKAGE = 'fb1e22aa70ad7f8d3f16c9f0a7c0a152d0e354f77177650d82da984b50ae6734'
SAVED_SOURCE_SHA = 'd512e5b2b6614b762d921dd94b2b5162687c0bbddde4caaf8b584b3b22dba745'
IDS = json.loads((HERE/'FIXED_IDS.json').read_bytes())['exact_ids']
FROOT = Path('F:/YananResearchStorage/GuardFed')

def need(ok, message):
    if not ok: raise ValueError(message)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()

def read(path): return json.loads(Path(path).read_bytes())

def save(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')

def digest_arg(value):
    need(re.fullmatch('[0-9a-f]{64}', value) is not None, 'Actual SHA256 required')
    return value

def volume(required=1024**3):
    v = json.loads(subprocess.check_output(['powershell', '-NoProfile', '-Command',
        'Get-Volume -DriveLetter F | Select-Object FileSystemLabel,HealthStatus,SizeRemaining | ConvertTo-Json -Compress'], text=True))
    need(v['FileSystemLabel']=='Yanan 2TB' and v['HealthStatus']=='Healthy' and v['SizeRemaining']>required, 'Required healthy F volume/capacity unavailable')
    return v

def fpath(path):
    p = Path(path).resolve()
    need(p.is_relative_to(FROOT.resolve()), 'Bulk evidence must be under GuardFed F storage')
    return p

def linux_proof(value):
    need(value['status']=='LINUX_ORIGINAL_FLGMM13_WHOLE_SAVED_CHECK_PASS_NOT_ROOT_ADOPTED', 'Whole Linux check has not passed')
    need([r['id'] for r in value['records']]==IDS and len(set(IDS))==13, 'Exact13 order/membership differs')
    need(value['original_check_saved_sha256']==SAVED_SOURCE_SHA, 'Original whole checker differs')
    need(value['package_sha256']==PACKAGE, 'Whole Linux source package differs')
    digest_arg(value['gate_result_sha256'])
    need(all(r['root_receipt_exact'] and r['cached_root_fit_exact'] and r['saved_predictions_metrics_counts_exact'] for r in value['records']), 'Incomplete whole saved proof')
    need(value['cached_root_refits']==13 and value['new_CNN']==value['new_training']==0 and value['test'] is False, 'Scientific scope differs')

def gate_proof(value):
    need(value['status']=='FLGMM_EXACT13_THREE_VIEW_PASS_NOT_ROOT_ADOPTED' and value['package_sha256']==PACKAGE, 'Wrong/unfinished gate')
    need([r['id'] for r in value['receipts']]==IDS, 'Gate exact13 differs')

def utc(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
