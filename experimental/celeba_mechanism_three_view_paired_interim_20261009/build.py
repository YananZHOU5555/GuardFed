"""Render a single accepted six-scene snapshot using the unchanged original statistics."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.dont_write_bytecode=True
import inputs as source

HERE=Path(__file__).resolve().parent
JOIN_SHA='8c67ee29de18ef136d4bdba58c3b82318b371aeb461e93c389961c6d6abc5701'
JOIN=source.ROOT/'tmp/celeba_valid_gpu_remaining440_v2_evidence_20261009/BOUNDED_DELTA_022_023_HANDOFF.json'
PANELS=[('All 10 seeds',list(range(91001,91011))),('Exclude selection seed: 9 seeds',list(range(91002,91011))),('Seeds 91005–91010: 6 seeds',list(range(91005,91011)))]
DISCLOSURES=[
    'Validation-only terminal checkpoints. This snapshot covers six complete Full–minus_U scenes, not all900 mechanism records or final test.',
    'ACC is percent; AEOD is absolute TPR gap, not full equalized odds; ASPD is absolute positive-rate gap. Mean ± sample SD uses ddof=1. Differences are minus_U minus Full; ACC differences are percentage points.',
    'Native includes each procedure’s original root-fitted calibration. Raw uses margin>0 (ties predict0); shared calibration uses the unchanged common root-only fit and >= group thresholds. These are three views of each same checkpoint, not three independent experiments.',
    'For all120 displayed checkpoint records, native and shared-calibration saved metrics and confusion counts are identical. Their equality in this subset supplies no independent calibration-gain evidence; it does not select the final reporting endpoint.',
    'Historical Full100 training used98 cu128 and2 cu130 records; current minus_U training used cu128/driver595. The displayed60 Full records include59 cu128 and1 cu130 (non-IID Benign91001). PyTorch-build and driver differences limit causal attribution.',
    'Full replay mixes CPU/GPU; minus_U replay uses CPU. Per-ID runtime/source provenance is retained. This is an implementation/numerical validation snapshot, not a uniform-device final fairness comparison.',
    'Seed91001 participated in recipe selection. Removing91001 or retaining91005–91010 applies equally to both procedures. All these validation seeds were previously exposed; neither panel is a prospectively untouched confirmation set.',
    'The unchanged loader materialized full-split Smiling/Male metadata during original replays. No test image inference, test fitting or test selection is performed here; this local extraction reads no label arrays.',
    'Completed scenes are included by coverage, independent of which variant wins. Improvements and regressions after deletion are retained. No significance test, Pareto claim, seed selection or claim that every component is necessary is made.',
    'Native/shared main endpoint remains pending user choice. Other mechanism variants and four remaining non-IID scenes are incomplete in this frozen60-control snapshot; later native71 records are excluded.'
]


def save(path,value):
    with Path(path).open('x',encoding='utf-8') as f:f.write(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def panels(records, original):
    expected=[('IID',a) for a in original.ATTACKS]+[('non-IID','Benign')]
    output=[];coverage={};paired={}
    for view in source.VIEWS:
        flat=[{k:r[k] for k in ('id','variant','distribution','attack','seed')} |
            dict(accuracy_pct=100*r['views'][view]['accuracy'],aeod=r['views'][view]['aeod'],aspd=r['views'][view]['aspd']) for r in records]
        summary=original.summarize(flat)
        complete=[(r['distribution'],r['attack']) for r in summary['paired_per_scene'] if r['variant']=='minus_U' and r['complete']]
        source.need(complete==expected,'Only the exact six complete ten-shared-seed scenes are publishable')
        coverage[view]=[{k:r[k] for k in ('variant','distribution','attack','n','expected_n','complete','seeds')} for r in summary['paired_per_scene'] if r['variant']=='minus_U']
        paired[view]=summary['paired_per_seed']
        for label,seeds in PANELS:
            rows=[]
            for dist,attack in complete:
                for variant in ('Full','minus_U','minus_U minus Full'):
                    candidates=summary['paired_per_seed'] if variant.endswith(' minus Full') else flat
                    selected=[r for r in candidates if r['distribution']==dist and r['attack']==attack and r['seed'] in seeds
                        and r['variant']==('minus_U' if variant.endswith(' minus Full') else variant)]
                    source.need({r['seed'] for r in selected}==set(seeds),'Matched seed panel incomplete')
                    rows.append(dict(distribution=dist,attack=attack,variant=variant,**original.statistic(selected,len(seeds))))
            output.append(dict(view=view,label=label,seeds=seeds,rows=rows))
    return output,coverage,paired


def markdown(result):
    text=['# CelebA Full–minus_U: interim three-view paired validation tables','',
        'Six coverage-complete scenes; all three views are reported in parallel. This snapshot does not select a primary endpoint.','']
    for panel in result['panels']:
        text += ['## '+panel['view']+' — '+panel['label'],'',
            '| Distribution | Scenario | Procedure / paired difference | n | ACC (%) ↑ | AEOD ↓ | ASPD ↓ |',
            '|---|---|---|---:|---:|---:|---:|']
        for row in panel['rows']:
            cells=[f"{row[m]['mean']:.3f} ± {row[m]['sample_sd_ddof1']:.3f}" if m=='accuracy_pct'
                else f"{row[m]['mean']:.5f} ± {row[m]['sample_sd_ddof1']:.5f}" for m in ('accuracy_pct','aeod','aspd')]
            text.append('| '+ ' | '.join([row['distribution'],row['attack'],row['variant'],str(row['n']),*cells])+' |')
        text += ['']
    text += ['## Evidence and limits','']+[p+'\n' for p in DISCLOSURES]
    text += ['Full identity join SHA256: `'+JOIN_SHA+'`.',
        'Original statistics source SHA256: `3d78db7e67275dce2d52bc8927dd86fc1c517708224a994345b70cadc56427ef`.','']
    return '\n'.join(text)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    source.need(args.output.resolve().parent==HERE and not args.output.exists(),'Fresh output only inside owned snapshot directory')
    bridge,inventory,baseline,original=source.context()
    controls=source.mechanism(bridge,inventory);full=source.full(bridge,inventory,baseline,JOIN,JOIN_SHA)
    matched={r['paired_full']['id'] for r in inventory['records']}
    selected=[r for r in full if r['id'] in matched]+controls
    source.need(len(selected)==120 and len({r['id'] for r in selected})==120,'Exact60 pairs required')
    full_by_cell={bridge.cell(r):r for r in selected if r['variant']=='Full'}
    pairs=[]
    for control in controls:
        reference=full_by_cell[bridge.cell(control)]
        source.need(control['data_contract']==reference['data_contract'],'Paired root/train/valid/client partition mismatch')
        pairs.append(dict(distribution=control['distribution'],attack=control['attack'],seed=control['seed'],Full=reference['id'],minus_U=control['id'],
            full_checkpoint_sha256=reference['checkpoint_sha256'],minus_U_checkpoint_sha256=control['checkpoint_sha256'],
            root_image_ids_sha256=control['data_contract']['root_image_ids_sha256'],data_contract_sha256=bridge.canonical(control['data_contract'])))
    result_panels,coverage,paired=panels(selected,original)
    negative=[]
    for panel in result_panels:
        if len(panel['seeds'])!=10:continue
        for row in panel['rows']:
            if row['variant']=='minus_U minus Full':
                negative.append({k:row[k] for k in ('distribution','attack')} | dict(view=panel['view'],
                    delta_ACC_pp=row['accuracy_pct']['mean'],delta_AEOD=row['aeod']['mean'],delta_ASPD=row['aspd']['mean'],
                    lower_disparity_after_deletion=[m for m in ('aeod','aspd') if row[m]['mean']<0],accuracy_increased_after_deletion=row['accuracy_pct']['mean']>0))
    result=dict(status='INTERIM_SIX_COMPLETE_SCENES_THREE_VIEWS_PENDING_ROOT_REVIEW',complete_scenes=6,paired_checkpoints=60,
        table_model_records=120,views=list(source.VIEWS),seed_panels=[dict(label=l,seeds=s) for l,s in PANELS],panels=result_panels,
        Full100_identity_available=True,baseline_collector_n=724,baseline_collector_sha256='456ceac1a9b149fea39ef4c91e7910d679a91123a02058106de105e0303b23c2',
        full_join_sha256=JOIN_SHA,mechanism_snapshot_n=60,scope_does_not_include_later_native_records=True,
        replay_devices={variant:dict(Counter(r['replay_runtime']['device'] for r in selected if r['variant']==variant)) for variant in ('Full','minus_U')},
        training_torch={variant:dict(Counter(r['training_torch'] for r in selected if r['variant']==variant)) for variant in ('Full','minus_U')},
        disclosures=DISCLOSURES,negative_and_positive_outcomes_retained=negative,primary_endpoint_selected=False,uniform_device_comparison=False,
        mechanism900_complete=False,final_test=False,new_inference=0,new_training=0,root_registration=False)
    args.output.mkdir()
    save(args.output/'tables.json',result);save(args.output/'records.json',dict(records=selected,pairs=pairs))
    save(args.output/'paired_per_seed.json',paired);save(args.output/'coverage.json',coverage)
    save(args.output/'INPUTS_SHA256.json',dict(files=source.PINS,source_statistics_unchanged=True,receipt_hashes_retained_per_record=True))
    with (args.output/'TABLES.md').open('x',encoding='utf-8') as f:f.write(markdown(result))
    print(json.dumps(dict(status=result['status'],scenes=6,views=3,panels=9,summary_rows=162,paired_checkpoints=60,output=str(args.output))))


if __name__=='__main__':main()
