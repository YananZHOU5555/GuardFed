"""Offline historical terminal-number candidate; no experiment or model loading."""
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
from statistics import mean
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import fitz

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
SOURCE=ROOT/'docs/server_deployment_20260923/training_20260923/synthetic_figure_recovery_20261009'
METRICS=['accuracy','aeod','aspd']
METHODS=['none','ctgan','forest_diffusion','gaussian_copula','pca_gaussian','smote','tvae']
LABELS={'none':'10% real baseline','ctgan':'CTGAN','forest_diffusion':'ForestDiffusion [1]','gaussian_copula':'Gaussian Copula','pca_gaussian':'PCA-Gaussian [2]','smote':'SMOTE','tvae':'TVAE'}
COLORS={'none':'#151515','ctgan':'#B88000','forest_diffusion':'#008D79','gaussian_copula':'#D55E00','pca_gaussian':'#9D5E9E','smote':'#637E09','tvae':'#007DA5'}
MARKERS=dict(zip(METHODS,['*','o','^','s','D','v','p']))

def sha(raw):return hashlib.sha256(raw).hexdigest()
def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(name,value):
    (HERE/name).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
def write_csv(name,rows):
    with (HERE/name).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

for name,pin in load(HERE/'INPUTS.json')['inputs'].items():
    raw=(ROOT/name).read_bytes();assert sha(raw)==pin['sha256'] and len(raw)==pin['bytes']
seal=load(SOURCE/'FILES_SHA256.json')
# Source package seal format follows the original audit; do not re-run its search.
items={row['path']:row for row in seal['files']}
assert len(items)==len(seal['files'])
for name in ['server_generation_original_260.jsonl','per_record_260.csv']:
    assert sha((SOURCE/name).read_bytes())==items[name]['sha256']
with (SOURCE/'per_record_260.csv').open(encoding='utf-8-sig',newline='') as f:csvrows=list(csv.DictReader(f))
byid={r['run_id']:r for r in csvrows};assert len(byid)==len(csvrows)==260
lines=(SOURCE/'server_generation_original_260.jsonl').read_bytes().splitlines(keepends=True)
assert len(lines)==260
records=[];rawrecords=[]
for i,line in enumerate(lines,1):
    r=json.loads(line);rawrecords.append(r);c=r['config'];old=byid[r['run_id']]
    assert sha(line)==old['raw_line_sha256']
    assert r['method']=='GuardFed-AD2+' and r['seed']==c['seed']==123 and r['rounds']==c['rounds']==70
    assert c['experiment_suite']=='server_generation_ablation'
    assert [t['round'] for t in r['last10_metrics']]==list(range(61,71))
    assert r['metrics']==r['last10_metrics'][-1]['metrics']
    assert r['dataset']==old['dataset'] and r['distribution']==old['distribution'] and r['attack']==old['attack']
    assert r['alpha']==float(old['alpha']) and c['experiment_tag']==old['tag']
    assert c['synthetic_method']==old['synthetic_method'] and c['server_ratio']==float(old['server_ratio']) and c['synthetic_ratio']==float(old['synthetic_ratio'])
    assert sha(json.dumps(c,sort_keys=True,separators=(',',':')).encode())==old['config_sha256_canonical']
    assert tuple(c[k] for k in ['server_ratio','synthetic_ratio']) in [(0.1,0.0),(0.01,0.09),(0.05,0.05)]
    for k,col in zip(METRICS,['terminal_ACC','terminal_AEOD','terminal_ASPD']):
        assert r['metrics'][k]==float(old[col]) and math.isfinite(r['metrics'][k]) and 0<=r['metrics'][k]<=1
    records.append({'subset_line':i,'original_raw_line':int(old['raw_line']),'raw_line_sha256':sha(line),'run_id':r['run_id'],'config_canonical_sha256':old['config_sha256_canonical'],
        'dataset':r['dataset'],'distribution':r['distribution'],'alpha':r['alpha'],'attack':r['attack'],'seed':123,'round':70,'historical_method_label':c['synthetic_method'],'tag':c['experiment_tag'],
        'real_root_ratio':c['server_ratio'],'synthetic_root_ratio':c['synthetic_ratio'],'accuracy':r['metrics']['accuracy'],'aeod':r['metrics']['aeod'],'aspd':r['metrics']['aspd'],
        'metric_source':'metrics == last10_metrics[round70]','checkpoint_binary_SHA_available':False})
assert len({r['run_id'] for r in records})==260
settings=sorted({(r['dataset'],r['tag']) for r in records})
assert len(settings)==26
points=[];lineage=[];differences=[]
expected=set(itertools.product(['IID','non-IID'],['Benign','F Flip','FedSA','S-DFA','Sp-DFA']))
for dataset,tag in settings:
    rs=[r for r in records if (r['dataset'],r['tag'])==(dataset,tag)]
    assert len(rs)==10 and {(r['distribution'],r['attack']) for r in rs}==expected
    assert len({(r['historical_method_label'],r['real_root_ratio'],r['synthetic_root_ratio']) for r in rs})==1
    first=rs[0];point_id=f'{dataset}:{tag}'
    vals={k:mean(r[k] for r in rs) for k in METRICS}
    original=[r for r in rawrecords if r['dataset']==dataset and r['config']['experiment_tag']==tag]
    arr=np.asarray([[r['last10_metrics'][-1]['metrics'][k] for k in METRICS] for r in original],dtype=float)
    assert arr.shape==(10,3)
    differences.extend(abs(arr.mean(axis=0)[n]-vals[k]) for n,k in enumerate(METRICS))
    points.append({'point_id':point_id,'dataset':dataset,'tag':tag,'historical_method_label':first['historical_method_label'],'real_root_ratio':first['real_root_ratio'],'synthetic_root_ratio':first['synthetic_root_ratio'],
        'round':70,'seed':123,'independent_seed_n':1,'equal_weight_scenarios':10,**vals,'accuracy_percent':100*vals['accuracy'],'source_subset_lines':';'.join(str(r['subset_line']) for r in rs)})
    lineage.extend({'point_id':point_id,'subset_line':r['subset_line'],'raw_line_sha256':r['raw_line_sha256'],'run_id':r['run_id']} for r in rs)
assert len(differences)==78 and max(differences)<1e-12
for dataset in ['adult','compas']:
    ps=[p for p in points if p['dataset']==dataset];assert len(ps)==13
    assert sum(p['historical_method_label']=='none' for p in ps)==1
    for m in METHODS[1:]:assert {(p['real_root_ratio'],p['synthetic_root_ratio']) for p in ps if p['historical_method_label']==m}=={(.01,.09),(.05,.05)}
write_csv('terminal_records_260.csv',records);write_csv('terminal_points_26.csv',points);write_csv('point_lineage_260.csv',lineage)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.labelsize':11,'axes.titlesize':12,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})
fig,axes=plt.subplots(2,2,figsize=(12,10))
fig.subplots_adjust(left=.087,right=.975,bottom=.185,top=.772,hspace=.43,wspace=.27)
fig.text(.5,.975,'AUTHOR-REVIEW CANDIDATE',ha='center',va='top',fontsize=15,weight='bold',color='#704321')
fig.text(.5,.943,'HISTORICAL terminal round 70 | n = 1 seed (123) | 10-scenario equal-weight means',ha='center',va='top',fontsize=11)
methodhandles=[Line2D([],[],marker=MARKERS[m],linestyle='None',color=COLORS[m],markerfacecolor=COLORS[m],markeredgewidth=1.4,markersize=9,label=LABELS[m]) for m in METHODS]
fig.legend(handles=methodhandles,loc='upper center',bbox_to_anchor=(.5,.909),ncol=4,frameon=False,columnspacing=1.5,handletextpad=.5,fontsize=10)
ratiohandles=[Line2D([],[],marker='o',linestyle='None',color='#555555',markerfacecolor='#555555',markersize=8,label='Filled: 1% real + 9% synthetic'),Line2D([],[],marker='o',linestyle='None',color='#555555',markerfacecolor='white',markeredgewidth=1.5,markersize=8,label='Open: 5% real + 5% synthetic'),Line2D([],[],marker='*',linestyle='None',color='#151515',markersize=11,label='Star: 10% real only')]
fig.legend(handles=ratiohandles,loc='upper center',bbox_to_anchor=(.5,.834),ncol=3,frameon=False,fontsize=9.5,columnspacing=1.5)
for row,dataset in enumerate(['adult','compas']):
    ps=[p for p in points if p['dataset']==dataset]
    for col,metric in enumerate(['aeod','aspd']):
        ax=axes[row,col]
        for point in sorted(ps,key=lambda x:x['historical_method_label']=='none'):
            m=point['historical_method_label'];filled=point['real_root_ratio']!=.05
            ax.scatter(point['accuracy_percent'],point[metric],s=135 if m=='none' else 75,marker=MARKERS[m],facecolors=COLORS[m] if filled else 'white',edgecolors=COLORS[m],linewidths=1.7,zorder=4 if m=='none' else 3)
        ax.set_title(f"({chr(97+row*2+col)}) {dataset.upper() if dataset=='compas' else 'Adult'}: ACC vs {metric.upper()}",loc='left',pad=10)
        ax.set_xlabel('Accuracy (%) - higher is better')
        ax.set_ylabel(metric.upper()+' - lower is better')
        ymax=max(p[metric] for p in ps);ax.set_ylim(-.03*ymax,1.10*ymax)
        xs=[p['accuracy_percent'] for p in ps];pad=max(.4,(max(xs)-min(xs))*.08);ax.set_xlim(min(xs)-pad,max(xs)+pad)
        ax.grid(True,color='#d9dde0',linestyle=':',linewidth=.7);ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)
fig.text(.087,.121,'[1] ForestDiffusion execution identity was not recovered. [2] Archived PCA-labelled code has no explicit PCA.',fontsize=9,va='top')
fig.text(.087,.096,'Historical method/ratio labels; test results were visible. Same-round metrics only: checkpoint binary SHA unavailable.',fontsize=9,va='top')
fig.text(.087,.071,'All 13 settings per dataset retained. Ten scenarios are not independent seeds; no SD/CI. No FairScore or round selection.',fontsize=9,va='top')
for fmt in ['png','svg','pdf']:
    fig.savefig(HERE/f'fig3_terminal_candidate.{fmt}',dpi=200,facecolor='white')
plt.close(fig)
doc=fitz.open(HERE/'fig3_terminal_candidate.pdf');assert len(doc)==1
page=doc[0];text=page.get_text()
for token in ['AUTHOR-REVIEW CANDIDATE','HISTORICAL terminal','n = 1','ForestDiffusion','checkpoint binary SHA','no SD/CI']:
    assert token in text,token
page.get_pixmap(matrix=fitz.Matrix(1.5,1.5)).save(HERE/'pdf_visual_check.png')
save('verification.json',{'status':'TERMINAL_NUMERICAL_CANDIDATE_PASS_PENDING_AUTHOR_ADOPTION','original_lines_SHA_checked':260,'unique_run_ids':260,'CSV_terminal_exact_triplets':260,'metrics_equal_last10_round70':260,'datasets':2,'settings_per_dataset':13,'aggregate_points':26,'scenarios_per_point':10,'independent_seed_n':1,'seed':123,'independent_numpy_mean_scalars':78,'max_abs_mean_difference':max(differences),'no_SD_CI':True,'no_joint_or_extrema_selection':True,'checkpoint_binary_identity_available':False,'history_test_visible':True,'new_science_execution':False,'original_figure_replaced':False,'pdf_pages':1,'pdf_text_required_notices_pass':True,'visual_review':'PNG and PDF render require visual inspection before final seal.'})
print(json.dumps({'points':26,'max_mean_diff':max(differences),'pdf_pages':1}))
