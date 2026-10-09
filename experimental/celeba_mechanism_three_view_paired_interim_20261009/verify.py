"""Independent table arithmetic and receipt identity checks; no metrics/model recomputation."""
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import tarfile
import tempfile

sys.dont_write_bytecode=True
import build
import inputs as source

HERE=Path(__file__).resolve().parent


def matches_receipt(record,receipt):
    source.need(record['id']==receipt['id'] and record['seed']==receipt['seed']
        and record['checkpoint_sha256']==receipt['checkpoint_sha256']
        and record['views']==receipt['views'] and record['fits']==receipt['fits'], 'Mixed saved checkpoint/views')


def main():
    folder=HERE/'snapshot_724_full100_mechanism60'
    read=lambda p:json.loads(p.read_bytes())
    tables=read(folder/'tables.json');data=read(folder/'records.json');paired=read(folder/'paired_per_seed.json');records=data['records']
    bridge,inventory,baseline,original=source.context()
    actual={r['id']:r for r in inventory['records']}|baseline
    need=source.need;errors=[];tar_cache={}
    try:
        for r in records:
            provenance=r['provenance'];identity=r['id']
            if 'receipt_path' in provenance:
                raw=Path(provenance['receipt_path']).read_bytes()
            else:
                archive=source.ROOT/provenance.get('archive',provenance.get('archive_path'))
                if archive not in tar_cache:tar_cache[archive]=tarfile.open(archive)
                raw=tar_cache[archive].extractfile(provenance['receipt_member']).read()
            need(hashlib.sha256(raw).hexdigest()==provenance['receipt_sha256'],'Actual scientific receipt SHA differs')
            receipt=json.loads(raw);matches_receipt(r,receipt)
            need(r['checkpoint_sha256']==actual[identity]['checkpoint']['sha256'] and r['config_sha256']==actual[identity]['config_canonical_sha256'],'Table model/config identity differs')
        by_cell={(r['variant'],r['distribution'],r['attack'],r['seed']):r for r in records}
        def value(r,view,metric):return r['views'][view]['accuracy']*100 if metric=='accuracy_pct' else r['views'][view][metric]
        cells=0
        markdown_rows=[line for line in (folder/'TABLES.md').read_text(encoding='utf-8').splitlines() if line.startswith('| ') and ' ± ' in line]
        rows=[(p,r) for p in tables['panels'] for r in p['rows']];need(len(rows)==len(markdown_rows)==162,'Markdown/table row coverage differs')
        for (panel,row),line in zip(rows,markdown_rows):
            for metric,index in [('accuracy_pct',4),('aeod',5),('aspd',6)]:
                def get(variant,seed):return by_cell[variant,row['distribution'],row['attack'],seed]
                values=[value(get('minus_U',seed),panel['view'],metric)-value(get('Full',seed),panel['view'],metric)
                    if row['variant']=='minus_U minus Full' else value(get(row['variant'],seed),panel['view'],metric) for seed in panel['seeds']]
                mean=math.fsum(values)/len(values);sd=math.sqrt(math.fsum((x-mean)**2 for x in values)/(len(values)-1))
                errors.extend([abs(mean-row[metric]['mean']),abs(sd-row[metric]['sample_sd_ddof1'])])
                need(max(errors[-2:])<=1e-12,'Independent table mean/sample SD mismatch')
                precision=3 if metric=='accuracy_pct' else 5
                cell=line.strip('| ').split(' | ')[index]
                need(cell==f'{row[metric]["mean"]:.{precision}f} ± {row[metric]["sample_sd_ddof1"]:.{precision}f}','Markdown numeric cell differs')
                cells+=1
        pair_checks=0
        for view,items in paired.items():
            for row in items:
                left=by_cell['minus_U',row['distribution'],row['attack'],row['seed']];right=by_cell['Full',row['distribution'],row['attack'],row['seed']]
                for metric in original.METRICS:
                    need(row[metric]==value(left,view,metric)-value(right,view,metric),'Per-seed pairing differs');pair_checks+=1
        refusals=[]
        def reject(name,call):
            try:call()
            except (ValueError,KeyError):refusals.append(name)
            else:raise AssertionError('Unexpected acceptance '+name)
        reject('incomplete_shared_seed',lambda:build.panels(records[1:],original))
        reject('duplicate_cell',lambda:build.panels(records+[records[0]],original))
        for key,change in [('root_ID',lambda inv:inv['records'][0]['data_contract'].__setitem__('root_image_ids_sha256','0'*64)),
            ('alpha',lambda inv:inv['records'][0]['config'].__setitem__('client_alpha',5.0)),
            ('tolerance',lambda inv:inv.__setitem__('native_tolerance',1e-9)),
            ('future_control',lambda inv:inv['records'].append(copy.deepcopy(inv['records'][0])))]:
            bad=copy.deepcopy(inventory);change(bad)
            reject(key,lambda bad=bad:bridge.validate_inventory(bad,{'records':list(baseline.values())}))
        sample=records[0];prov=sample['provenance']
        raw=tar_cache[source.ROOT/prov['archive_path']].extractfile(prov['receipt_member']).read() if 'archive_path' in prov else Path(prov['receipt_path']).read_bytes()
        good=json.loads(raw)
        for key,change in [('checkpoint',lambda r:r.__setitem__('checkpoint_sha256','0'*64)),
            ('seed',lambda r:r.__setitem__('seed',91010)),('missing_view',lambda r:r['views'].pop('raw')),
            ('mixed_shared_view',lambda r:r['views']['shared_calibration'].__setitem__('accuracy',0.123))]:
            bad=copy.deepcopy(good);change(bad);reject(key,lambda bad=bad:matches_receipt(sample,bad))
        reject('external_join_SHA_drift',lambda:source.read(build.JOIN,'0'*64))
        join=read(build.JOIN)
        with tempfile.TemporaryDirectory(prefix='_refusal_',dir=HERE) as temporary:
            fixture=Path(temporary);need(fixture.resolve().parent==HERE.resolve(),'Fixture outside owned directory')
            for name,change in [('missing_Full',lambda j:j['Full100_identity_coverage']['one_to_one_mapping'].pop()),
                ('duplicate_Full',lambda j:j['Full100_identity_coverage']['one_to_one_mapping'].__setitem__(1,copy.deepcopy(j['Full100_identity_coverage']['one_to_one_mapping'][0]))),
                ('Full_receipt_SHA',lambda j:j['Full100_identity_coverage']['one_to_one_mapping'][0]['accepted_evidence'].__setitem__('receipt_sha256','0'*64))]:
                bad=copy.deepcopy(join);change(bad);path=fixture/(name+'.json');path.write_text(json.dumps(bad),encoding='utf-8')
                reject(name,lambda path=path:source.full(bridge,inventory,baseline,path,source.sha(path)))
        identical=sum(r['views']['native']==r['views']['shared_calibration'] for r in records)
        report=dict(status='PASS_INDEPENDENT_TABLE_NUMERICS_AND_SOURCE_REFUSALS',actual_receipt_id_checkpoint_three_views_checked=len(records),
            mean_SD_numeric_checks=len(errors),Markdown_numeric_cells=cells,paired_per_seed_metric_checks=pair_checks,
            max_abs_independent_statistics_difference=max(errors),absolute_statistics_check_tolerance=1e-12,
            original_native_acceptance_tolerance_unchanged=1e-12,native_shared_saved_metrics_and_counts_identical_records=identical,
            rejection_count=len(refusals),refusals=refusals,torch_imported='torch' in sys.modules,label_arrays_read=False,new_CNN=False,new_training=False,network=False)
        build.save(folder/'NUMERIC_CHECKS.json',report)
        print(json.dumps(report,indent=2))
    finally:
        for bundle in tar_cache.values():bundle.close()


if __name__=='__main__':main()
