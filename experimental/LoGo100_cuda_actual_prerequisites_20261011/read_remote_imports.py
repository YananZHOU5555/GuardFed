from pathlib import Path
import hashlib,json,subprocess,ast
assert hashlib.sha256(Path('/etc/vast-agents-guide.md').read_bytes()).hexdigest()=='42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa'
p=Path('/workspace/GuardFed-celeba-expanded/src/data_loader.py');s=p.read_text();tree=ast.parse(s)
imports=[ast.get_source_segment(s,n) for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom))]
q=subprocess.run(['/usr/local/bin/uv','python','list','--only-installed','--offline'],capture_output=True,text=True,timeout=20)
print(json.dumps({'data_loader':{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'top_level_imports':imports},'uv_installed_pythons':{'returncode':q.returncode,'stdout':q.stdout,'stderr':q.stderr}}))
