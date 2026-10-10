"""Check the user's external evidence volume before any new bulk write."""
from pathlib import Path
import json
import subprocess

STORAGE_ROOT = Path('F:/YananResearchStorage/GuardFed')
RESERVE_BYTES = 1024 ** 3


def check_bulk_storage(required_bytes=0):
    command = (
        "$ErrorActionPreference='Stop'; Get-Volume -DriveLetter F | "
        "Select-Object DriveLetter,FileSystemLabel,SizeRemaining,HealthStatus | "
        "ConvertTo-Json -Compress"
    )
    result = subprocess.run(
        ['powershell', '-NoProfile', '-NonInteractive', '-Command', command],
        check=True, capture_output=True, text=True, timeout=30,
    )
    volume = json.loads(result.stdout.lstrip('\ufeff'))
    assert volume['DriveLetter'] == 'F' and volume['FileSystemLabel'] == 'Yanan 2TB', 'Required Yanan 2TB volume is unavailable; no internal-drive fallback'
    assert volume['HealthStatus'] in ('Healthy', 0), 'External volume is not healthy'
    assert required_bytes >= 0 and volume['SizeRemaining'] >= required_bytes + RESERVE_BYTES, 'External volume has insufficient space; preserve server evidence'
    assert STORAGE_ROOT.resolve().drive.upper() == 'F:', 'Bulk destination resolves outside F'
    return dict(volume=volume, required_bytes=required_bytes, reserve_bytes=RESERVE_BYTES,
                local_bulk_root=STORAGE_ROOT.as_posix(), fallback_to_internal_drive=False)
