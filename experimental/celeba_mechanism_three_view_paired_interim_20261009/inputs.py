"""Read already accepted/offserver receipts only; never load models or label arrays."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tarfile

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
PREP = ROOT/'tmp/celeba_mechanism_valid_incremental_next37_20261009'
VIEWS = ('native', 'raw', 'shared_calibration')
PINS = {}


def need(ok, message):
    if not ok: raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024), b''):h.update(block)
    return h.hexdigest()


def read(path, expected=None):
    path=Path(path);actual=sha(path)
    if expected is not None:need(actual==expected, 'Source/input drift: '+str(path))
    PINS[str(path)]=actual
    return json.loads(path.read_bytes())


def module(name, path, expected):
    need(sha(path)==expected, 'Scientific source drift')
    PINS[str(path)]=expected
    spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec)
    sys.modules[name]=mod;spec.loader.exec_module(mod)
    return mod


def context():
    bridge=module('original_next37_bridge',PREP/'bridge.py','a1eba316c68f99b6527330ba0fb5b7b113b2a0ea9f33410bc23331e12a5f03c9')
    inventory=read(PREP/'inventory_actual60_Full100refs.json','0f837e22a1dd316fadbae85464f2dedbf074fcd5de4c96a3c974f426c7ee55d7')
    baseline=read(ROOT/'tmp/celeba_final_valid_replay_20261009/inputs/model_inventory.json',bridge.BASELINE_INVENTORY_SHA)
    bridge.validate_inventory(inventory,baseline)
    original=module('original_mechanism_statistics',ROOT/'tmp/celeba_mechanism_evidence_20261009/evidence_v4.py',bridge.EVIDENCE_V4_SHA)
    return bridge,inventory,{r['id']:r for r in baseline['records']},original


def receipt_identity(receipt, record, bridge):
    need(all(receipt[k]==record[k] for k in ('id','method','distribution','attack','seed','config_canonical_sha256')), 'Receipt/cell/config mismatch')
    need(receipt['checkpoint_sha256']==record['checkpoint']['sha256'] and receipt['original_result_sha256']==record['result']['sha256']
        and receipt['original_job_sha256']==record['raw_job']['sha256'], 'Mixed checkpoint/result/rawjob')
    need(receipt['model_inventory_record_sha256']==bridge.canonical(record), 'Original scientific record mismatch')
    need(receipt['valid_n']==19867 and receipt['root_reconstruction']['root_n']==16277
        and receipt['valid_image_ids_sha256']==record['data_contract']['evaluation_image_ids_sha256']
        and receipt['root_reconstruction']['root_image_ids_sha256']==record['data_contract']['root_image_ids_sha256'], 'Root/valid IDs differ')
    need(receipt['weights_before']==receipt['weights_after'] and not receipt['optimizer_created'] and not receipt['gradients_created'], 'Inference-only/weight identity failed')
    need(set(receipt['views'])==set(receipt['fits'])==set(VIEWS), 'All three views from the same receipt required')
    for view in VIEWS:
        fit=receipt['fits'][view]
        need(fit['method']==record['method'] and fit['view']==view, 'View/method drift')
        need(fit['fit_data']==('none' if view=='raw' else 'clean_train_root_only'), 'Fit split drift')
    need(receipt['native_comparison']['accepted'] and receipt['native_comparison']['max_abs_difference']<=1e-12, 'Native tolerance changed/failed')
    need(max(abs(receipt['views']['native'][m]-record['prior_validation_metrics'][m]) for m in ('accuracy','aeod','aspd'))<=1e-12, 'Original native metrics differ')


def normalized(receipt, record, variant, provenance):
    return dict(id=record['id'],variant=variant,method=record['method'],distribution=record['distribution'],attack=record['attack'],seed=record['seed'],
        checkpoint_sha256=record['checkpoint']['sha256'],config_sha256=record['config_canonical_sha256'],data_contract=record['data_contract'],
        training_torch=record['training_torch'],replay_runtime=receipt['runtime'],views=receipt['views'],fits=receipt['fits'],provenance=provenance)


def mechanism(bridge, inventory):
    actual={r['id']:r for r in inventory['records']};specs=list(inventory['closed_replay_evidence']['records'])
    for rel,expected in inventory['closed_replay_evidence']['input_pins'].items():read_or_hash=ROOT/rel;need(sha(read_or_hash)==expected,'Prior23 proof/source drift');PINS[str(read_or_hash)]=expected
    # Two externally adopted increments are the only extension beyond original23.
    parent=PREP/'execution_candidate/backups'
    previous=None
    for tag,receipt_sha,proof_sha,adoption_sha in [
        ('incremental_20261009T141304Z','e22525155441aa6d2368b252ca103fdba03978c00e45659e1ba72123c1a45ec6','5f8e6c13e490c113af244ee24eb94b528a0721d1e0949812a22aa37b74ea5247','48b219abe106d4edfb2a8bf1425928c712fcdac7919ed3f1c5e2f688ac9e1955'),
        ('incremental_20261009T144055Z','2450df47666aae4ffac2bab7cdb21479599c7a4bd77619273f1395860df7189f','cb85a457cd4cfe7a4e08000003c58e6491f9b09237fab8bb983edcc2e85f73b9','a7a563390de455914b33cf62064d674347e0c26754a778b9a344ac99bdc4fac3')]:
        folder=parent/tag;r=read(folder/'backup_receipt.json',receipt_sha);p=read(folder/'OFFSERVER_VERIFICATION.json',proof_sha);a=read(folder/'ROOT_ADOPTION_REVIEW.json',adoption_sha)
        need(r['previous_backup_receipt_sha256']==previous and r['accepted_new_ids']==p['accepted_new_ids']==a['accepted_new_ids'], 'Next37 adopted increment chain differs')
        archive=folder/'incremental_valid_three_views.tar.gz'
        need(sha(archive)==r['archive_sha256']==p['archive_sha256'], 'Next37 archive/offserver drift')
        for identity in r['accepted_new_ids']:
            specs.append(dict(id=identity,archive=str(archive),archive_sha256=r['archive_sha256'],strict_acceptance_member='runs/'+identity+'/strict_acceptance.json',offserver_proof_sha256=proof_sha,root_adoption_sha256=adoption_sha))
        previous=receipt_sha
    need(len(specs)==len({r['id'] for r in specs})==60 and {r['id'] for r in specs}==set(actual), 'Exact accepted60 coverage changed')
    old_paths={
        'be4d1d34b21443572a25e6a189710e8d75b108a8fcbdb58896a71f2ba1d88cc8':ROOT/'tmp/celeba_mechanism_valid_replay_20261009/inventory_actual8_Full100refs.json',
        '288d2afb260f7eb77bcccba82e7edf6dbfe0519cd5bcc42fb546496236eeadbc':ROOT/'tmp/celeba_mechanism_valid_incremental_v2_20261009/inventory_actual23_Full100refs.json',
        sha(PREP/'inventory_actual60_Full100refs.json'):PREP/'inventory_actual60_Full100refs.json'}
    originals={key:{r['id']:r for r in read(path,key)['records']} for key,path in old_paths.items()}
    result=[]
    for archive_name in dict.fromkeys(r['archive'] for r in specs):
        group=[r for r in specs if r['archive']==archive_name];archive=ROOT/archive_name
        need(sha(archive)==group[0]['archive_sha256'], 'Accepted archive changed');PINS[str(archive)]=group[0]['archive_sha256']
        with tarfile.open(archive) as bundle:
            need(len(bundle.getnames())==len(set(bundle.getnames())), 'Duplicate archive member')
            for item in group:
                identity=item['id'];raw=bundle.extractfile(item['strict_acceptance_member']).read();strict=json.loads(raw)
                if 'strict_acceptance_sha256' in item:need(hashlib.sha256(raw).hexdigest()==item['strict_acceptance_sha256'],'Original strict member drift')
                def member(name):
                    matches=[n for n in bundle.getnames() if n.endswith('/'+identity+'/'+name)]
                    need(len(matches)==1,'Ambiguous saved scientific receipt')
                    return matches[0],bundle.extractfile(matches[0]).read()
                receipt_member,raw_receipt=member('receipt.json');_,raw_bridge=member('bridge_receipt.json')
                receipt=json.loads(raw_receipt);proof=json.loads(raw_bridge)
                need(strict['status']=='MECHANISM_VALID_THREE_VIEWS_ACCEPTED' and strict['id']==identity and strict['views']==receipt['views'], 'Original strict views differ')
                need(strict['bridge_receipt_sha256']==hashlib.sha256(raw_bridge).hexdigest() and proof['scientific_body_receipt_sha256']==hashlib.sha256(raw_receipt).hexdigest(),'Strict/bridge/receipt chain differs')
                need(proof['source_before']==proof['source_after'] and proof['artifact_before']==proof['artifact_after'],'Original sources/artifacts changed')
                record=originals[strict['inventory_sha256']][identity];current=actual[identity]
                for key in ('config','source_hashes','adapter_source_hashes','data_contract','prior_validation_metrics','paired_full','runtime_paths'):
                    need(record[key]==current[key], 'Prior/current scientific identity differs: '+key)
                receipt_identity(receipt,record,bridge)
                need(strict['checkpoint_sha256']==current['checkpoint']['sha256'] and strict['paired_full_reference']==current['paired_full'],'Wrong paired Full/checkpoint')
                provenance=dict(item,receipt_member=receipt_member,receipt_sha256=hashlib.sha256(raw_receipt).hexdigest(),strict_acceptance_sha256=hashlib.sha256(raw).hexdigest(),inventory_sha256=strict['inventory_sha256'],bridge_receipt_sha256=hashlib.sha256(raw_bridge).hexdigest())
                result.append(normalized(receipt,current,'minus_U',provenance))
    return result


def full(bridge, inventory, baseline, handoff_path, handoff_sha):
    handoff=read(handoff_path,handoff_sha);coverage=handoff['Full100_identity_coverage']
    need(coverage['status']=='FULL100_EXACT_IDENTITY_COVERAGE_AND_SAVED_RECEIPT_SOURCE_JOIN_PASS'
        and coverage['accepted_at_finish724_n']==100 and coverage['missing_at_finish724_ids']==[], 'Full100 actual identity coverage missing')
    mapping=coverage['one_to_one_mapping'];refs={r['id']:r for r in inventory['full_references']}
    need(len(mapping)==len({r['original_full_reference_id'] for r in mapping})==100
        and {r['original_full_reference_id'] for r in mapping}==set(refs), 'Duplicate/missing Full identity mapping')
    chain=coverage['collector_chain_verified']
    need(chain[0]['collector_sha256']=='456ceac1a9b149fea39ef4c91e7910d679a91123a02058106de105e0303b23c2','Different frozen724 snapshot')
    for link in chain:
        c=read(link['collector_path'],link['collector_sha256'])
        for key in ('previous_collector','previous_424_collector'):
            if key+'_path' in link:read(link[key+'_path'],link[key+'_sha256'])
        if 'proof_path' in link:read(link['proof_path'],link['proof_sha256'])
        need(len(c['accepted_ids'])==len(set(c['accepted_ids']))==c['accepted_n'],'Collector duplicate/count drift')
    accepted=set(read(chain[0]['collector_path'],chain[0]['collector_sha256'])['accepted_ids'])
    for authority in coverage['mechanism_authority_sources_verified']:read_or_hash=Path(authority['path']);need(sha(read_or_hash)==authority['sha256'],'Original mechanism authority drift');PINS[str(read_or_hash)]=authority['sha256']
    result=[]
    for item in mapping:
        identity=item['baseline_inventory_id'];record=baseline[identity];ref=refs[identity];e=item['accepted_evidence']
        need(item['original_full_reference_id']==item['accepted_collector_id']==identity and identity in accepted,'Full collector/reference identity differs')
        need(bridge.full_reference(record)==ref and bridge.canonical(record)==item['baseline_record_canonical_sha256']
            and bridge.canonical(ref)==item['full_reference_canonical_sha256'],'Full reference/canonical record mismatch')
        need(item['pairing_key']=={k:record[k] for k in ('distribution','attack','seed')} and item['method']==item['source_method']==record['method']=='GuardFed-AD2+','Full method/cell mismatch')
        proof=read(e['offserver_proof_path'],e['offserver_proof_sha256'])
        if 'receipt_path' in e:
            receipt=read(e['receipt_path'],e['receipt_sha256']);strict=read(e['strict_acceptance_path'],e['strict_acceptance_sha256'])
            need(sha(e['prediction_array_path'])==e['prediction_arrays_sha256'],'Full saved array SHA mismatch');PINS[e['prediction_array_path']]=e['prediction_arrays_sha256']
            matches=[r for r in proof['results'] if r['id']==identity]
            need(proof['status']=='ROOT_GPU440_RESOURCE_GUARD_V2_CHUNK_OFFSERVER_SAVED_ARRAY_PASS' and len(matches)==1
                and matches[0]['receipt_sha256']==e['receipt_sha256'] and matches[0]['array_sha256']==e['prediction_arrays_sha256']
                and proof['strict_sha256']==e['strict_acceptance_sha256'] and strict['scope']==proof['recovery_scope'],'Full GPU original strict/offserver chain differs')
        else:
            archive=Path(e['archive_path']);need(sha(archive)==e['archive_sha256']==proof['archive_sha256'],'Full CPU archive/offserver mismatch');PINS[str(archive)]=e['archive_sha256']
            origin=read(e['origin_collector_path'],e['origin_collector_sha256']);need(identity in origin['accepted_ids'],'Full CPU original collector missing')
            with tarfile.open(archive) as bundle:
                need(len(bundle.getnames())==len(set(bundle.getnames())),'Duplicate Full archive member')
                raw=bundle.extractfile(e['receipt_member']).read();need(hashlib.sha256(raw).hexdigest()==e['receipt_sha256'],'Full CPU receipt member drift');receipt=json.loads(raw)
                raw=bundle.extractfile(e['strict_acceptance_member']).read();need(hashlib.sha256(raw).hexdigest()==e['strict_acceptance_sha256'],'Full CPU strict member drift');strict=json.loads(raw)
                need(hashlib.sha256(bundle.extractfile(e['prediction_array_member']).read()).hexdigest()==e['prediction_arrays_sha256'],'Full CPU saved array member drift')
            need(proof['status']=='PASS' and proof.get('strict_acceptance_sha256',proof.get('run_receipt_sha256'))==e['strict_acceptance_sha256'],'Full CPU original strict/offserver SHA differs')
        need(strict['status'] in ('SELECTED_NATIVE_VALID_REPLAY_PASS','SELECTED_VALID_REPLAY_ACCEPTED') and identity in strict['accepted_ids'],'Full original strict is incomplete')
        need(receipt['prediction_arrays_sha256']==e['prediction_arrays_sha256'],'Full receipt/array binding differs')
        receipt_identity(receipt,record,bridge)
        result.append(normalized(receipt,record,'Full',e))
    return result
