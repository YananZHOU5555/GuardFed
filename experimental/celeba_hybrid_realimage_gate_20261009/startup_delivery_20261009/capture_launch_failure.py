"""Bounded launcher-only diagnostic; preserved failure precedes all science."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
STAGE = HERE.parent
TARGET = HERE / 'launcher_failure_v1'
assert not TARGET.exists() and not (STAGE / 'execution_repair_v1/runtime_overlay').exists()
script = Path('/opt/supervisor-scripts/guardfed_celeba_hybrid_writer_repair.sh')
config = Path('/etc/supervisor/conf.d/guardfed_celeba_hybrid_writer_repair.conf')
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
before = {str(p): digest(p) for p in [script, config, HERE / 'APPROVED.json', STAGE / 'execution_repair_v1/FILES_SHA256.json']}
result = subprocess.run(['/bin/bash', str(script)], capture_output=True, text=True,
    env={'PATH': '/usr/bin:/bin:/opt/instance-tools/bin', 'PROC_NAME': 'guardfed_celeba_hybrid_writer_repair', 'WORKSPACE': '/workspace'}, timeout=20)
assert result.returncode != 0 and '$1: unbound variable' in result.stderr
assert not (STAGE / 'execution_repair_v1/runtime_overlay').exists() and not (HERE / 'run.log').exists()
assert before == {name: digest(Path(name)) for name in before}
TARGET.mkdir()
for p in (script, config):
    (TARGET / p.name).write_bytes(p.read_bytes())
proof = dict(status='TERMINAL_LAUNCHER_FAILURE_BEFORE_PYTHON', argv=['/bin/bash', str(script)],
    diagnostic_returncode=result.returncode, stdout=result.stdout, stderr=result.stderr,
    scientific_runtime_directory_exists=False, gate_log_exists=False, new_prediction_or_model_files=0,
    scientific_module_imports_performed=False, source_approval_and_launcher_sha_before_after_identical=before,
    logging_tool_sha256=digest(Path('/opt/supervisor-scripts/utils/logging.sh')),
    cause='set -u makes logging.sh logpath=$1 fail before logging redirection and before environment/Python',
    diagnostic_executed_only_to_capture_same_pre_science_shell_error=True,
    root_authorized_recovery='One independent launcher v2 only; shared utils initialize without nounset; scientific9/APPROVED unchanged; no automatic retry')
(TARGET / 'failure.json').write_bytes((json.dumps(proof, indent=2) + '\n').encode())
(TARGET / 'FILES_SHA256.json').write_bytes((json.dumps({'files': {p.name: digest(p) for p in TARGET.iterdir() if p.is_file()}}, indent=2) + '\n').encode())
print(json.dumps(proof), flush=True)
