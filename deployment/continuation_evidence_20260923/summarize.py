#!/usr/bin/env python3
"""Independent acceptance and terminal-round statistics for continuation cohorts."""
import collections
import csv
import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics

OUT=Path(__file__).resolve().parent
QUEUES=[('GuardFed-next','compas_root_noise_v1'),('GuardFed-rootshare','rootshare_v1')]
METRICS=['accuracy','aeod','aspd']
KEYS=['queue','source_kind','dataset','distribution','alpha','method','attack','ablation','root_label_noise','root_sensitive_noise','root_protected_share','preprocessing']


def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False)
def sha_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def signature(config):
    return hashlib.sha256(canonical({k:v for k,v in config.items() if k not in ['seed','experiment_suite','experiment_tag']}).encode()).hexdigest()
def group_info(queue,source,obj):
    c=obj['config'];alpha=c.get('client_alpha')
    if alpha is None:alpha={'IID':5000.0,'non-IID':5.0}[obj['distribution']]
    return dict(zip(KEYS,[queue,source,obj['dataset'],obj['distribution'],float(alpha),obj['method'],obj['attack'],c.get('ablation_component','none'),c.get('root_label_noise',0.0),c.get('root_sensitive_noise',0.0),c.get('root_protected_share',None),c.get('compas_preprocessing_version','legacy') if obj['dataset']=='compas' else 'adult_original_train_fit']))
def key(info):return tuple(info[k] for k in KEYS)
def write_csv(name,rows):
    columns=list(dict.fromkeys(k for row in rows for k in row))
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows(rows)


def main():
    started=datetime.datetime.now(datetime.timezone.utc).isoformat()
    queues=[];histories=[];needed=collections.defaultdict(set);manifest_seeds={}
    for home,name in QUEUES:
        root=Path('/workspace')/home;folder=root/'results/revision_20260923'/name
        mp=folder/'manifest.json';m=json.loads(mp.read_text());manifest_seeds[name]=set(m['seeds']);jobs=[json.loads(Path(x).read_text()) for x in m['jobs']]
        history=[]
        for p in sorted(folder.glob('reused_full/*.json')):
            w=json.loads(p.read_text());history.append((p,w))
            if 'source' in w:needed[w['source']].add(w['source_line'])
        queues.append((root,name,folder,m,jobs,history,sha_file(mp)))
    historical_source={}
    for source,numbers in needed.items():
        with open(source,'rb') as f:
            for n,line in enumerate(f,1):
                if n in numbers:historical_source[(source,n)]=(hashlib.sha256(line).hexdigest(),json.loads(line))
                if n>=max(numbers):break
    accepted=[];audits=[];issues=[];qrows=[];expected=collections.defaultdict(list);run_ids=collections.defaultdict(set)
    for root,name,folder,m,jobs,history,manifest_sha in queues:
        queue_issues=[];seen_ids=set();sourcechecks={};new_accept=hist_accept=0;missing=invalid=0
        for rel,digest in m['source_hashes'].items():
            p=root/rel
            sourcechecks[rel]={'expected':digest,'current_sha256':sha_file(p) if p.is_file() else None}
            sourcechecks[rel]['matches_current']=sourcechecks[rel]['current_sha256']==digest
            if not sourcechecks[rel]['matches_current']:queue_issues.append('Current source hash mismatch: '+rel)
        for job in jobs:
            if job['id'] in seen_ids:queue_issues.append('Duplicate manifest job id: '+job['id'])
            seen_ids.add(job['id'])
        entries=[('new',Path(j['output'])/'result.json',j,None) for j in jobs]
        entries += [('historical',p,None,w) for p,w in history]
        for kind,path,job,wrapper in entries:
            obj=job if job is not None else wrapper['result'];info=group_info(name,kind,obj);gkey=key(info);seed=obj['config']['seed']
            expected[gkey].append(seed)
            audit={**info,'seed':seed,'file':str(path),'status':'missing','errors':'','checkpoint_status':'not_available','checkpoint_sha256':None}
            if not path.exists():missing+=1;audits.append(audit);continue
            errors=[]
            try:
                payload=path.read_bytes();d=json.loads(payload) if kind=='new' else wrapper['result'];c=d['config']
                assert d['rounds']==70 and c['rounds']==70,'round count must be 70'
                assert d['seed']==seed==c['seed'],'seed mismatch'
                assert d['run_id'] not in run_ids[name],'duplicate run_id within queue'
                run_ids[name].add(d['run_id'])
                assert group_info(name,kind,d)==info,'condition mismatch'
                assert float(d['alpha'])==info['alpha'],'alpha mismatch'
                rounds=d['trajectory_metrics'];assert [r['round'] for r in rounds]==list(range(1,71)),'metric trajectory not exactly rounds1..70'
                assert canonical(d['metrics'])==canonical(rounds[-1]['metrics']),'reported metrics differ from final-round metrics'
                dc=d['data_contract'];assert not dc['feature_includes_label'] and not dc['feature_includes_sensitive'],'label/sensitive leakage flag'
                if d['dataset']=='compas':assert dc.get('preprocessing_version')=='train_only' and c.get('compas_preprocessing_version')=='train_only','COMPAS must use train_only'
                root_audit=dc['root_noise_audit'];root_rows=root_audit['root_rows']
                assert sum(root_audit['clean_group_label_counts'].values())==root_rows,'clean root support count mismatch'
                assert sum(root_audit['observed_group_label_counts'].values())==root_rows,'observed root support count mismatch'
                for field,audit_key in [('root_label_noise','label'),('root_sensitive_noise','sensitive')]:
                    rate=c.get(field,0.0);details=root_audit[audit_key]
                    assert details['requested_rate']==rate,'root requested noise mismatch'
                    assert details['flipped_count']==math.floor(rate*root_rows),'root actual noise count mismatch'
                assert all(canonical(r['root_noise_audit'])==canonical(root_audit) for r in d['round_summaries']),'root diagnostics changed within run'
                for diagnostic in d['round_summaries']:
                    weights=diagnostic['aggregate']['client_weights']
                    assert len(weights)==c['num_clients'] and abs(sum(weights)-1)<1e-6,'aggregation weights mismatch'
                    assert all(math.isfinite(w) and w>=0 for w in weights),'invalid aggregation weight'
                if c.get('root_protected_share') is not None:
                    share=dc['server_sampling_audit'];protected={'adult':0,'compas':1}[d['dataset']]
                    assert share['target_protected_share']==c['root_protected_share'],'share target mismatch'
                    assert share['protected_group_value']==protected,'protected coding mismatch'
                    assert share['server_sensitive_counts'][str(protected)]==round(root_rows*c['root_protected_share']),'protected count mismatch'
                    assert share['actual_protected_share']==share['server_sensitive_counts'][str(protected)]/root_rows,'actual protected share mismatch'
                    assert share['client_rows']+share['reserve_rows']==dc['train_rows'],'reservoir/client partition size mismatch'
                if kind=='new':
                    rj=d['revision_job'];assert rj['id']==job['id'],'embedded job id mismatch'
                    assert canonical(c)==canonical(job['config'])==canonical(rj['config']),'manifest/result/job config mismatch'
                    assert rj['evidence_stage']==job['evidence_stage']=='formal_supplement','not formal evidence stage'
                    assert rj['source_hashes']==job['source_hashes'],'runtime source hashes differ from job'
                    assert all(job['source_hashes'].get(k)==v for k,v in m['source_hashes'].items()),'job source hashes differ from manifest'
                    assert [r['round'] for r in d['round_summaries']]==list(range(1,71)),'new round diagnostics not complete'
                    checkpoint=path.parent/'model.pt';actual=sha_file(checkpoint)
                    audit['checkpoint_sha256']=actual;assert actual==rj['checkpoint_sha256'],'checkpoint SHA256 mismatch'
                    audit['checkpoint_status']='sha256_verified'
                    environment=f"new_{c['device']};torch={rj['torch_version']};python={rj['python_version'].split()[0]}"
                else:
                    original_path=Path(wrapper['source_result'])
                    assert sha_file(original_path)==wrapper['source_result_sha256'],'reused result SHA256 mismatch'
                    original=json.loads(original_path.read_text())
                    assert canonical(original)==canonical(d),'reused wrapper differs from original result'
                    source_job=json.loads(Path(wrapper['source_job']).read_text())
                    rj=d['revision_job']
                    assert canonical(source_job['config'])==canonical(c)==canonical(rj['config']),'reused source config mismatch'
                    assert rj['id']==source_job['id'],'reused source job id mismatch'
                    assert rj['source_hashes']==source_job['source_hashes'],'reused source hashes mismatch'
                    assert [r['round'] for r in d['round_summaries']]==list(range(1,71)),'reused diagnostics not complete'
                    checkpoint=original_path.parent/'model.pt';actual=sha_file(checkpoint)
                    assert actual==rj['checkpoint_sha256'],'reused checkpoint SHA256 mismatch'
                    audit['checkpoint_status']='reused_checkpoint_sha256_verified'
                    audit['checkpoint_sha256']=actual
                    environment=f"reused_{c['device']};torch={rj['torch_version']};python={rj['python_version'].split()[0]}"
                for metric in METRICS:
                    value=d['metrics'][metric]
                    if value is not None and math.isfinite(value):assert 0<=value<=1,f'{metric} out of range'
                row={**info,'seed':seed,'run_id':d['run_id'],'round':70,**d['metrics'],
                    'result_file':str(path),'result_sha256':hashlib.sha256(payload).hexdigest(),
                    'config_sha256':hashlib.sha256(canonical(c).encode()).hexdigest(),'condition_config_sha256':signature(c),
                    'checkpoint_status':audit['checkpoint_status'],'checkpoint_sha256':audit['checkpoint_sha256'],
                    'environment':environment,'warnings':canonical(d.get('warnings',[])),
                    'evaluation_stats':canonical(d.get('evaluation_stats',{})),
                    'root_noise_audit':canonical(dc.get('root_noise_audit',{})),
                    'root_sampling_audit':canonical(dc.get('server_sampling_audit',{})),
                    'reused_source':wrapper['source_result'] if kind=='historical' else '',
                    'reused_source_line':'',
                    'reused_source_sha256':wrapper['source_result_sha256'] if kind=='historical' else ''}
                accepted.append(row);audit['status']='accepted';new_accept+=kind=='new';hist_accept+=kind=='historical'
            except Exception as exc:
                errors.append(f'{type(exc).__name__}: {exc}');audit['status']='invalid';invalid+=1
                issues.append({'queue':name,'file':str(path),'errors':errors})
            audit['errors']='; '.join(errors);audits.append(audit)
        expected_paths={str(Path(j['output'])/'result.json') for j in jobs}
        extra=[str(p) for p in folder.glob('runs/*/result.json') if str(p) not in expected_paths]
        if extra:issues.append({'queue':name,'unexpected_results_not_pooled':extra})
        if len(history)!=m.get('reused_full_count',0):queue_issues.append('Historical wrapper count differs from manifest')
        if len(jobs)!=m['new_run_count']:queue_issues.append('Job count differs from manifest')
        if len(m['seeds'])!=10 or len(set(m['seeds']))!=10:queue_issues.append('Manifest does not contain 10 unique seeds')
        if queue_issues:issues.append({'queue':name,'manifest_errors':queue_issues})
        qrows.append({'queue':name,'manifest_file':str(folder/'manifest.json'),'manifest_sha256':manifest_sha,
            'new_expected':len(jobs),'new_accepted':new_accept,'historical_expected':m.get('reused_full_count',0),'historical_accepted':hist_accept,
            'missing':missing,'invalid':invalid,'unexpected_result_files':len(extra),
            'status':'invalid' if invalid or queue_issues or extra else 'partial' if missing else 'complete',
            'runtime_source_hashes_checked':True,'current_source_files':sourcechecks})
        print(json.dumps({k:v for k,v in qrows[-1].items() if k not in ['current_source_files','manifest_file','manifest_sha256']}),flush=True)
    invariant_groups=collections.defaultdict(list)
    for row in accepted:
        invariant_groups[(row['queue'],row['dataset'],row['seed'])].append(row)
    invariant_checks=[]
    for context,rows in invariant_groups.items():
        audits=[json.loads(row['root_sampling_audit']) for row in rows]
        if context[0]=='rootshare_v1':
            fields=['server_rows','reserve_index_sha256','client_index_sha256','client_rows']
            assert all(len({a[field] for a in audits})==1 for field in fields),('rootshare invariant failed',context)
            invariant_checks.append({'queue':context[0],'dataset':context[1],'seed':context[2],'accepted_runs':len(rows),'fixed_root_clients_reservoir':'verified'})
        else:
            root_audits=[json.loads(row['root_noise_audit']) for row in rows]
            assert len({a['clean_root_sha256'] for a in root_audits})==1,('root noise baseline changed',context)
            assert len({canonical(a['clean_group_label_counts']) for a in root_audits})==1,('clean root support changed',context)
            invariant_checks.append({'queue':context[0],'dataset':context[1],'seed':context[2],'accepted_runs':len(rows),'fixed_clean_root':'verified'})
    (OUT/'root_invariant_checks.json').write_text(json.dumps(invariant_checks,indent=2)+'\n')
    grouped=collections.defaultdict(list)
    for r in accepted:grouped[key(r)].append(r)
    coverage=[];formal=[];metric_coverage=[]
    for gkey,seeds in sorted(expected.items()):
        rows=grouped[gkey];info=dict(zip(KEYS,gkey));actual=[r['seed'] for r in rows];fingerprints={r['condition_config_sha256'] for r in rows}
        valid= set(seeds)==manifest_seeds[info['queue']] and len(seeds)==len(set(seeds))==10 and len(actual)==len(set(actual))==10 and set(actual)==set(seeds) and len(fingerprints)==1
        status='complete_10seed' if valid else 'partial' if len(actual)<10 else 'invalid_seed_or_config_set'
        if len(fingerprints)>1:
            status='mixed_configs';issues.append({**info,'condition_config_hashes':sorted(fingerprints)})
        cov={**info,'status':status,'expected_seed_n':len(seeds),'accepted_seed_n':len(actual),'expected_seeds':canonical(sorted(seeds)),
            'accepted_seeds':canonical(sorted(actual)),'missing_seeds':canonical(sorted(set(seeds)-set(actual))),'condition_config_n':len(fingerprints)}
        coverage.append(cov)
        for metric in METRICS:
            vals=[r[metric] for r in rows if r[metric] is not None and math.isfinite(r[metric])]
            mrow={**info,'metric':metric,'n_expected':10,'n_accepted':len(rows),'n_defined':len(vals),'status':status}
            if valid and len(vals)==10:
                mrow.update(mean=statistics.mean(vals),sample_std=statistics.stdev(vals),std_ddof=1)
                formal.append(mrow.copy())
            elif valid:mrow['status']='complete_10seed_but_metric_undefined'
            metric_coverage.append(mrow)
    # Duplicate historical controls across two queues are intentional references, not new independent experiments.
    unique=collections.Counter((r['reused_source'],r['reused_source_sha256']) for r in accepted if r['source_kind']=='historical')
    report={'snapshot_started_utc':started,'snapshot_completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'queues':qrows,'accepted_rows':len(accepted),'new_checkpoint_sha_verified_n':sum(r['source_kind']=='new' for r in accepted),
        'historical_rows':sum(unique.values()),'reused_checkpoint_sha_verified_n':sum(r['source_kind']=='historical' for r in accepted),'total_checkpoint_sha_verified_n':len(accepted),'unique_historical_source_records':len(unique),
        'duplicate_historical_references_across_queues':sum(v-1 for v in unique.values()),
        'complete_10seed_conditions':sum(r['status']=='complete_10seed' for r in coverage),'total_conditions':len(coverage),
        'formal_metric_rows':len(formal),'issues':issues,'warnings':[{k:r[k] for k in ['queue','seed','result_file','warnings']} for r in accepted if r['warnings']!='[]'],
        'limitations':[
            'Only manifest jobs and explicitly reused historical controls are included. Pipeline checks/pilots are not scientific evidence and are excluded.',
            'All metrics use round70, same reported checkpoint. No ranking, seed filtering, best-round selection or significance claims.',
            'Formal mean and sample standard deviation (ddof=1) require all ten unique planned seeds, one config per condition and ten defined values; incomplete/undefined groups remain visible without invented summaries.',
            'Reused COMPAS clean S-DFA controls are prior train_only CUDA runs with source result and checkpoint SHA256 verified. New CPU noise is not asserted 70-round numerically hardware-equivalent.',
            'COMPAS new experiments use train_only preprocessing. Old legacy metric values are not pooled and are not direct matched controls.',
            'Root-share is a new fixed-reservoir matched protocol, not a reuse of old main-table controls. At zero protected share, root fairness and thresholds for that group are unidentifiable; existing algorithm fallbacks are not evidence of root fairness.',
            'Root noise is end-to-end: both defense and the FedSA-inspired attack depend on root updates. It is not a defense-only intervention.',
            'Repeated historical references across queues are the same evidence, not additional independent samples.',
            'Current snapshot is based on results already written to disk. Prepared/running labels do not establish completion.']}
    write_csv('per_seed.csv',accepted);write_csv('run_acceptance.csv',audits);write_csv('condition_coverage.csv',coverage)
    write_csv('formal_mean_sample_std.csv',formal);write_csv('metric_coverage.csv',metric_coverage)
    (OUT/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    lookup={(key(r),r['metric']):r for r in metric_coverage}
    table_lines=['# 返修实验：第70轮均值与样本标准差','',
        '仅完整的10个计划seed且该指标全部有定义时报告 mean ± sample std（ddof=1）。数值采用0–1尺度；每项指标来自同一末轮模型。未完成条件不报告均值。',
        '历史控制单独标注；复用CUDA控制与本次CPU执行环境不等同，原始结果及checkpoint均已核验。COMPAS使用train_only预处理，不与旧legacy数值混合。pipeline检查和短测均不计入科学证据。',
        f"快照：{report['snapshot_completed_utc']}", '']
    for _,queue_name in QUEUES:
        table_lines += ['## '+queue_name,'',
            '| 来源 | 数据 | α | 方法 | 攻击 | 消融 | root标签/属性噪声 | root保护组占比 | 预处理 | seed数 | ACC | AEOD | ASPD |',
            '|---|---|---:|---|---|---|---|---|---|---:|---:|---:|---:|']
        for cov in coverage:
            if cov['queue']!=queue_name:continue
            fields=[]
            for metric in METRICS:
                row=lookup[(key(cov),metric)]
                fields.append(f"{row['mean']:.5f} ± {row['sample_std']:.5f}" if 'mean' in row else f"未汇总 ({row['n_defined']}/10有定义)")
            source='历史复用' if cov['source_kind']=='historical' else '新增'
            ablation='Full' if cov['ablation']=='none' else '-'+cov['ablation']
            table_lines.append(f"| {source} | {cov['dataset']} | {cov['alpha']:g} | {cov['method']} | {cov['attack']} | {ablation} | {cov['root_label_noise']:g}/{cov['root_sensitive_noise']:g} | {cov['root_protected_share']} | {cov['preprocessing']} | {cov['accepted_seed_n']}/10 | {' | '.join(fields)} |")
        table_lines.append('')
    table_lines += ['## 解释限制','']+['- '+v for v in report['limitations']]
    (OUT/'mean_sample_std.md').write_text('\n'.join(table_lines)+'\n')

    lines=['# Revision evidence: validated result entry point','',f"Snapshot: {started}",
        f"Accepted records: {len(accepted)}; new checkpoint SHA256 checks: {report['new_checkpoint_sha_verified_n']}; unique historical records: {len(unique)}.",
        f"Complete ten-seed conditions: {report['complete_10seed_conditions']}/{len(coverage)}; validation issues: {len(issues)}.",'',
        '| Queue | New accepted / expected | Historical accepted / expected | Status |','|---|---:|---:|---|']
    for q in qrows:lines.append(f"| {q['queue']} | {q['new_accepted']}/{q['new_expected']} | {q['historical_accepted']}/{q['historical_expected']} | {q['status']} |")
    lines+=['','- `mean_sample_std.md`: complete comparison tables grouped by queue; incomplete conditions remain labeled.',
        '- `formal_mean_sample_std.csv`: complete 10-seed conditions only, metric scale 0–1, ddof=1.',
        '- `per_seed.csv`: all accepted individual results, environment, provenance hashes and warning/support context.',
        '- `condition_coverage.csv` / `metric_coverage.csv`: missing seeds and undefined metrics remain visible.',
        '- `run_acceptance.csv` / `verification.json`: per-run checks, source/checkpoint validation, anomalies.',
        '- Root-share conditions use the fixed reservoir protocol and never borrow old main-table controls.',
        '- Incomplete cohorts remain partial; no mean is emitted for an incomplete ten-seed condition.',
        '- Rerun: `/workspace/GuardFed-next/.venv/bin/python deployment/continuation_evidence_20260923/summarize.py`.','']
    lines+=['- '+v for v in report['limitations']]
    (OUT/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:report[k] for k in ['accepted_rows','new_checkpoint_sha_verified_n','historical_rows','unique_historical_source_records','complete_10seed_conditions','total_conditions','formal_metric_rows','issues']},indent=2),flush=True)
    if issues:raise SystemExit(1)


if __name__=='__main__':main()
