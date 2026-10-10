"""Adopt actual closed32 after the separate original strict/prediction review."""
from pathlib import Path
import datetime, hashlib, json, math, sys

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
SOURCE=ROOT/'tmp/celeba_logofair_screen32_20261010'
BULK=Path('F:/YananResearchStorage/GuardFed/logofair_screen32_20261010/attempt001')
sys.path.insert(0,str(ROOT/'tmp'))
from guardfed_local_storage import check_bulk_storage
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())
storage=check_bulk_storage()
review=HERE/'COMPLETE32_REVIEW.json'
assert sha(review)=='70f8cf59f11d505bd5d6fd1f9eaf7e9f64be629349bfef3b8fbd995486c06d64'
p=read(review)
assert p['original_strict_rechecked_n']==32 and p['new_fits']==p['new_CNN_calls']==0 and not p['test_inference']
assert sha(SOURCE/'FILES_SHA256.json')=='accd5cb8582a344f870188f1e70661b6dc9dc948cc88c9e6f6651e451607bc49'
for n,h in read(SOURCE/'FILES_SHA256.json')['files'].items():assert sha(SOURCE/n)==h
summary=BULK/'SUMMARY32.json'; index=BULK/'STRICT32_INDEX.json'
assert sha(summary)==p['summary_sha256']=='c051c4ebbe22a43519bf4ede715d1f54093bf77baa4e5f613be36e9841eb74c6'
assert sha(index)==p['strict_index_sha256']
s=read(summary); ix=read(index); manifest=read(SOURCE/'jobs/manifest.json')
assert len(s['records'])==len(ix['records'])==len(manifest['jobs'])==32
assert {r['id'] for r in s['records']}=={r['id'] for r in ix['records']}=={r['id'] for r in manifest['jobs']}
expected={r['id']:r for r in manifest['jobs']}; reported={r['id']:r for r in s['records']}
constant=[]
for r in ix['records']:
    job=read(SOURCE/'jobs'/expected[r['id']]['job'])
    assert sha(SOURCE/'jobs'/expected[r['id']]['job'])==expected[r['id']]['job_sha256']
    assert sha(r['result'])==r['result_sha256'] and sha(r['acceptance'])==r['acceptance_sha256']
    result=read(r['result']); accept=read(r['acceptance'])
    assert result['job']==job and result['metrics']==reported[r['id']]['metrics']
    assert result['checkpoint_sha256']==reported[r['id']]['checkpoint_sha256']
    assert accept['status']=='PASS' and accept['job_sha256']==expected[r['id']]['job_sha256']
    assert job['seed']==91001 and job['fit_seed']==1719 and job['evaluation_split']=='valid'
    assert len(result['history'])==job['settings']['post_rounds']==30
    for n,h in accept['artifact_hashes'].items():assert sha(Path(r['result']).parent/n)==h
    assert result['metrics']['prediction_count']==19867
    if result['metrics']['positive_rate'] in (0.0,1.0):constant.append(r['id'])
def score(m):
    a,b=m['aeod'],m['aspd']; worst=max(a,b)
    return m['accuracy']-.35*(.45*a+.45*b+.10*worst)-.10*max(0,worst-.06)
computed=[]
for candidate in sorted({r['candidate'] for r in s['records']}):
    rows=[r for r in s['records'] if r['candidate']==candidate]
    assert len(rows)==4 and {(r['distribution'],r['attack']) for r in rows}=={('IID','Benign'),('IID','S-DFA'),('non-IID','Benign'),('non-IID','S-DFA')}
    computed.append(dict(candidate=candidate,score=math.fsum(score(r['metrics']) for r in rows)/4,
        **{k:math.fsum(r['metrics'][k] for r in rows)/4 for k in ('accuracy','aeod','aspd')}))
assert len(computed)==8 and max(abs(a[k]-b[k]) for a,b in zip(computed,s['candidates']) for k in ('accuracy','aeod','aspd','score'))<=1e-12
winner=min(computed,key=lambda r:(-r['score'],r['candidate']))
assert winner['candidate']==s['selected_per_method']['LoGoFair-DP-official-adapted']['candidate']==p['selected_candidate']
protocol=read(SOURCE/'snapshot/logofair_bridge_20261010/protocol.json')
candidate=next(r for r in protocol['candidates'] if r['id']==winner['candidate'])
proof=dict(status='ROOT_LOGOFAIR_SCREEN32_ADOPTED',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    accepted_count=32,accepted_ids=[r['id'] for r in ix['records']],
    summary_path=str(summary),summary_sha256=sha(summary),strict_index_path=str(index),strict_index_sha256=sha(index),
    source_seal_sha256=sha(SOURCE/'FILES_SHA256.json'),independent_acceptance_path=review.relative_to(ROOT).as_posix(),independent_acceptance_sha256=sha(review),
    selected_recipe=candidate,selected_candidate=winner,accuracy_champion=s['accuracy_champion'],three_metric_pareto=s['three_metric_pareto'],
    all_candidates_retained=True,constant_prediction_ids=constant,seed_n=1,fit_seed=1719,conditions_not_independent_seeds=True,
    sample_SD_reported=False,significance_claimed=False,virtual_cohorts=20,true_training_client_fairness=False,
    original_strict_and_saved_predictions_rechecked=32,root_source_and_artifact_SHA_checked=True,
    new_CNN_calls=0,new_fits=0,test_evaluated=False,fullcoverage100_started=False,storage_preflight=storage,
    limitations=p['limitations'])
out=HERE/'ROOT_ADOPTION.json'
with out.open('x',encoding='utf8') as f:f.write(json.dumps(proof,indent=2,allow_nan=False)+'\n')
lines=['# LoGoFair 验证搜索：完整32项已接受','',
    '8候选 × IID/non-IID × Benign/S-DFA；模型seed91001、拟合seed1719。原算法30轮后处理；不是70轮CNN新增训练。',
    '', '| 候选 | 四条件平均ACC (%) | AEOD | ASPD | 冻结score |','|---|---:|---:|---:|---:|']
for r in computed:lines.append(f"| {r['candidate']} | {100*r['accuracy']:.4f} | {r['aeod']:.6f} | {r['aspd']:.6f} | {r['score']:.9f} |")
lines+=['',f"冻结规则选择 {winner['candidate']}；准确率冠军 {s['accuracy_champion']['candidate']}。全部候选及{len(constant)}项恒定预测保留。",
    '', '单模型seed搜索；四场景不是四个独立seed，不计算样本SD/显著性。20个image-ID哈希虚拟cohort为项目人口适配，不代表真实训练client公平性；原AEOD为绝对TPR差。最终test未运行，完整100覆盖尚未启动。','']
with (HERE/'RESULTS.md').open('x',encoding='utf8') as f:f.write('\n'.join(lines))
print(json.dumps(dict(status=proof['status'],accepted=32,candidate=candidate,constant_prediction_count=len(constant),root_adoption_sha256=sha(out))))
