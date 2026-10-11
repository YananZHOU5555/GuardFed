#!/bin/sh
# PROPOSAL ONLY. Root must approve execution; no training/evaluation is called.
# Original caller rejects interpreter symlinks, so venv uses --copies.
set -eu
test "$(sha256sum /etc/vast-agents-guide.md | cut -d ' ' -f1)" = '42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa'
LoGo_ENV='/workspace/guardfed_envs/logofair-frozen-post-cpu310-20261011'
test ! -e "$LoGo_ENV"
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
# Managed Python is isolated from /venv/main and current CUDA queues.
export UV_PYTHON_INSTALL_DIR='/workspace/guardfed_envs/logofair-python-managed-20261011'
nice -n 10 ionice -c 3 /usr/local/bin/uv python install 3.10
LoGo_PYTHON=$(/usr/local/bin/uv python find 3.10)
nice -n 10 ionice -c 3 "$LoGo_PYTHON" -m venv --copies "$LoGo_ENV"
test -f "$LoGo_ENV/bin/python"
test ! -L "$LoGo_ENV/bin/python"
nice -n 10 ionice -c 3 /usr/local/bin/uv pip install --python "$LoGo_ENV/bin/python" --index-url https://download.pytorch.org/whl/cpu 'torch==2.8.0+cpu'
nice -n 10 ionice -c 3 /usr/local/bin/uv pip install --python "$LoGo_ENV/bin/python" 'numpy==1.26.4' 'netcal==1.3.6' 'scikit-learn==1.7.1' 'scipy==1.15.3' 'pandas==2.3.1' 'pyro-ppl==1.9.2' 'gpytorch==1.15.2'
# Metadata check only. Actual import/pickle/canary are separate root-authorized gates.
"$LoGo_ENV/bin/python" -B -c 'import sys,json,importlib.metadata as m; expected={"torch":"2.8.0+cpu","numpy":"1.26.4","netcal":"1.3.6","scikit-learn":"1.7.1","scipy":"1.15.3","pandas":"2.3.1","pyro-ppl":"1.9.2","gpytorch":"1.15.2"}; actual={n:m.version(n) for n in expected}; assert sys.version_info[:2]==(3,10) and actual==expected,(sys.version,actual); print(json.dumps({"python":sys.version,"executable":sys.executable,"packages":actual}))'
# Capture logs, executable/package hashes and actual Python patch version externally.
# Any install failure stops; do not reuse a partially created environment silently.
