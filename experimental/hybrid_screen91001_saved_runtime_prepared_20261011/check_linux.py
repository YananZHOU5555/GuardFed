"""Explicit later root call, once: unchanged whole saved checker on Linux, no CNN."""
from pathlib import Path
import argparse, hashlib, json, shlex, subprocess
from contract import HERE, digest_arg, linux_proof, need, read, save, sha, utc

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gate-result-sha256',required=True,type=digest_arg)
    p.add_argument('--report-dir',required=True,type=Path)
    p.add_argument('--post-replay-preflight',required=True)
    p.add_argument('--post-replay-preflight-sha256',required=True,type=digest_arg)
    p.add_argument('--allow-original-cached-root-refit',action='store_true',required=True)
    a=p.parse_args();need(__debug__,'Do not use -O')
    out=a.report_dir.resolve();need(out!=HERE and not out.is_relative_to(HERE),'Prepared source stays immutable')
    out.mkdir(parents=True,exist_ok=True)
    need(not any((out/n).exists() for n in ('LINUX_CHECK_COMMAND.json','LINUX_SAVED_CHECK.json','LINUX_CHECK_EXIT.json')),'Single attempt; preserve failure and stop')
    need(a.post_replay_preflight.startswith('/workspace/guardfed_checks/') and '\n' not in a.post_replay_preflight,'Actual Linux preflight path required')
    source=(HERE/'linux_saved_remote.py').read_text(encoding='utf-8')
    cmd='env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 taskset -c 110 ionice -c 3 nice -n 10 /workspace/guardfed_envs/celeba-cu128-20261009/bin/python -B -c '+shlex.quote(source)+' --gate-result-sha256 '+a.gate_result_sha256+' --post-replay-preflight '+shlex.quote(a.post_replay_preflight)+' --post-replay-preflight-sha256 '+a.post_replay_preflight_sha256+' --allow-original-cached-root-refit'
    save(out/'LINUX_CHECK_COMMAND.json',{'utc':utc(),'source_sha256':hashlib.sha256(source.encode()).hexdigest(),'gate_result_sha256':a.gate_result_sha256,'CPU':110,'threads':1,'post_replay_preflight':a.post_replay_preflight,'post_replay_preflight_sha256':a.post_replay_preflight_sha256,'package_sha256':'9be2ce96b4548144903e9e928567efa96565326917e2363f1ad11712ca44015d','allow_original_cached_root_refit':True,'single_attempt':True,'new_CNN':0,'new_training':0,'test':False})
    try:
        c=subprocess.run(['ssh','-p','60350','-o','BatchMode=yes','-o','ConnectTimeout=15','root@89.22.197.55',cmd],stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=7200)
    except subprocess.TimeoutExpired as e:
        (out/'LINUX_CHECK_STDOUT.txt').write_bytes(e.stdout or b'');(out/'LINUX_CHECK_STDERR.txt').write_bytes(e.stderr or b'')
        save(out/'LINUX_CHECK_EXIT.json',{'exit':None,'timeout':True,'retry_authorized':False,'remote_process_state':'unknown_do_not_resend'})
        raise
    (out/'LINUX_CHECK_STDOUT.txt').write_bytes(c.stdout);(out/'LINUX_CHECK_STDERR.txt').write_bytes(c.stderr)
    save(out/'LINUX_CHECK_EXIT.json',{'exit':c.returncode,'utc':utc()})
    need(c.returncode==0,'Original whole checker failed; preserve stderr and do not retry')
    proof=json.loads(c.stdout);linux_proof(proof)
    need(proof['gate_result_sha256']==a.gate_result_sha256,'Whole Linux proof/gate differs')
    (out/'LINUX_SAVED_CHECK.json').write_bytes(c.stdout)
    print(json.dumps({'status':proof['status'],'records':1,'linux_proof_sha256':sha(out/'LINUX_SAVED_CHECK.json')}))

if __name__=='__main__':main()
