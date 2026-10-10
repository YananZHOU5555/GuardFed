"""Fresh F-volume guard before downloading or restoring bulk; no internal-drive fallback."""
from pathlib import Path
import json,subprocess
BASE=Path(json.loads((Path(__file__).resolve().parent/'DELTA_SCOPE.json').read_bytes())['local_bulk_root'])
assert BASE.resolve().is_relative_to(Path('F:/YananResearchStorage/GuardFed').resolve())
def guard(archive,out):
    for p in (Path(archive),Path(out)):
        if not p.resolve().is_relative_to(BASE.resolve()):raise ValueError('Bulk must stay in dedicated F directory')
    r=subprocess.run(['powershell','-NoProfile','-Command',"Get-Volume -DriveLetter F | Select-Object FileSystemLabel,Size,SizeRemaining,HealthStatus | ConvertTo-Json -Compress"],capture_output=True,text=True,check=True)
    volume=json.loads(r.stdout)
    if volume['FileSystemLabel']!='Yanan 2TB' or volume['HealthStatus']!='Healthy':raise ValueError('Expected healthy Yanan 2TB F volume')
    # Selected closed CNN records plus frozen stage identity; conservative fixed free-space floor.
    if volume['SizeRemaining']<5*1024**3:raise ValueError('F free space below5GiB; no fallback')
    return volume
if __name__=='__main__':print(json.dumps(guard(BASE/'delta.tar.gz',BASE/'verified')))
