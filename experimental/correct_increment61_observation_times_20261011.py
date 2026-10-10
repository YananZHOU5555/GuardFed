"""Keep distinct live sampling times explicit; no acceptance changes."""
from pathlib import Path
import datetime,hashlib,json,os
R=Path(__file__).resolve().parents[1];T=R/'docs/server_deployment_20260923/training_20260923'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
p=R/'tmp/increment61_entry_update_proof_20261011.json';previous=json.loads(p.read_bytes())
sp=T/'TRAINING_STATE.json';assert sha(sp)==previous['STATE_sha256']=='f1a285b0253d8197df25727d7f00c9a6b2c017a9c5b118bdd27aa4acf0f88f4b'
s=json.loads(sp.read_bytes());proof=dict(previous);proof['status']='CURRENT_ENTRIES_ACTUAL_FL71_HYBRID10_DISTINCT_OBSERVATION_TIMES'
proof['previous_update_proof_sha256']=sha(p);proof['correction_source_sha256']=sha(__file__);proof['entries']={}
markers={'RUNNING.md':b'# HISTORICAL:','REBUTTAL_COMPLETION_20261009.md':b'## Historical accepted increment','返修实验总览.md':'以下为 2026-10-04'.encode(),'MONITOR_HANDOFF.md':b'# Historical handoff snapshots','EXECUTION.md':b'# HISTORICAL PREPARATION SNAPSHOT'}
for rel,pin in previous['entries'].items():
 q=R/rel;assert sha(q)==pin['sha256'];b=q.read_bytes();i=b.index(markers[q.name]);history=b[i:];prefix=b[:i].decode()
 changes={'最近实测（2026-10-10T21:15:50.604577+00:00，不代替验收）':'分批实测（UTC见格内，不代替验收）','完成328、活动8、等待464、失败0，活动第44–65轮':'21:15:50：完成328、活动8、等待464、失败0，活动第44–65轮','remaining620远端闭合147，与离机接受140单列':'20:39:34：remaining620远端闭合147，与离机接受140单列','终轮71；三视图接受71':'20:39:34：终轮71；三视图接受71','终轮55；未完成全部搜索或选择recipe':'20:39:34：终轮55；未完成全部搜索或选择recipe','终轮16；IID Benign十seed native及三视图结果已接纳':'20:39:34：终轮16；IID Benign十seed native及三视图结果已接纳','该实测双GPU利用率':'21:15:50资源实测：双GPU利用率'}
 for old,new in changes.items():assert prefix.count(old)==1,(q,old);prefix=prefix.replace(old,new)
 q.write_bytes(prefix.encode()+history);assert hashlib.sha256(history).hexdigest()==pin['history_sha256']
 proof['entries'][rel]=dict(sha256=sha(q),history_sha256=pin['history_sha256'],history_exact=True)
s['latest_five_queue_readonly_observation']['scope']='21:15 UTC main queue rounds plus five service/worker identities and resources only; auxiliary terminal counts remain separate 20:39 UTC observations.'
s['current_entry_correction']=dict(path=Path(__file__).relative_to(R).as_posix(),sha256=sha(__file__),reason='Distinct observation UTC values; accepted counts unchanged',utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
temp=sp.with_suffix('.times.tmp');assert not temp.exists();temp.write_text(json.dumps(s,ensure_ascii=False,indent=2)+'\n',encoding='utf8');os.replace(temp,sp)
proof['STATE_sha256']=sha(sp);out=R/'tmp/increment61_entry_times_corrected_proof_20261011.json';assert not out.exists();out.write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps({'STATE_sha256':sha(sp),'proof_sha256':sha(out)}))
