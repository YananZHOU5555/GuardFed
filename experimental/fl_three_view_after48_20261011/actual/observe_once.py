from pathlib import Path
import base64, datetime, hashlib, json, subprocess, sys
H=Path(__file__).resolve().parent
name=sys.argv[1] if len(sys.argv)>1 else 'OBSERVATION'
assert name in ('OBSERVATION','FRESH_OBSERVATION')
assert not (H/(name+'.json')).exists()
source=(H/'observer.py').read_bytes()
remote="import base64; exec(compile(base64.b64decode('"+base64.b64encode(source).decode()+"'),'<readonly-observer>','exec'))"
cmd=['ssh','-p','60350','-o','BatchMode=yes','-o','ConnectTimeout=20','root@89.22.197.55',"nice -n 10 ionice -c 3 python3 -c \""+remote+"\""]
start=datetime.datetime.now(datetime.timezone.utc).isoformat()
p=subprocess.run(cmd,input=(H/'observer_payload.json').read_bytes(),capture_output=True,timeout=300)
(H/(name+'.stdout')).write_bytes(p.stdout);(H/(name+'.stderr')).write_bytes(p.stderr)
(H/(name+'_COMMAND.json')).write_text(json.dumps({'argv':cmd,'started_utc':start,'exit':p.returncode,'observer_sha256':hashlib.sha256(source).hexdigest()},indent=2)+'\n')
assert p.returncode==0,p.stderr.decode(errors='replace')
data=json.loads(p.stdout)
(H/(name+'.json')).write_text(json.dumps(data,indent=2)+'\n')
print(json.dumps({'utc':data['utc'],'hashes':len(data['hashes']),'hashes_OK':data['source_model_data_hashes_verified'],'conflicts':len(data['narrow_cpu120_127_conflicts']),'selected_live':sum(len(x['live_producers']) for x in data['selected']),'selected_failure':sum(len(x['failures']) for x in data['selected']),'duplicate':len(data['duplicate_gate'])}))
