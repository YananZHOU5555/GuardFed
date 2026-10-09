"""Read-only mechanism identity bridge to sealed terminal validation replay.

No CLI dispatch, training, dataset load or inference runs on import. Runtime
functions require an externally SHA-approved, bounded CPU-only valid receipt.
"""
from __future__ import annotations
import copy
import hashlib
import importlib.util
import itertools
import json
import math
import os
from pathlib import Path, PurePosixPath
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
SCOPE = 'MECHANISM_TERMINAL_VALID_REPLAY_C_AFTER36'
ACCEPTED_IDS = ['minus_C_IID_Benign_seed91001', 'minus_C_IID_Benign_seed91002', 'minus_C_IID_Benign_seed91003', 'minus_C_IID_Benign_seed91004', 'minus_C_IID_Benign_seed91005', 'minus_C_IID_Benign_seed91006', 'minus_C_IID_Benign_seed91007', 'minus_C_IID_Benign_seed91008', 'minus_C_IID_Benign_seed91009', 'minus_C_IID_Benign_seed91010', 'minus_C_IID_F Flip_seed91001', 'minus_C_IID_F Flip_seed91002', 'minus_C_IID_F Flip_seed91003', 'minus_C_IID_F Flip_seed91004', 'minus_C_IID_F Flip_seed91005', 'minus_C_IID_F Flip_seed91006', 'minus_C_IID_F Flip_seed91007', 'minus_C_IID_F Flip_seed91008', 'minus_C_IID_F Flip_seed91009', 'minus_C_IID_F Flip_seed91010', 'minus_C_IID_FedSA_seed91001', 'minus_C_IID_FedSA_seed91002', 'minus_C_IID_FedSA_seed91003', 'minus_C_IID_FedSA_seed91004', 'minus_C_IID_FedSA_seed91005', 'minus_C_IID_FedSA_seed91006', 'minus_C_IID_FedSA_seed91007', 'minus_C_IID_FedSA_seed91008', 'minus_C_IID_FedSA_seed91009', 'minus_C_IID_FedSA_seed91010', 'minus_C_IID_S-DFA_seed91001', 'minus_C_IID_S-DFA_seed91002', 'minus_C_IID_S-DFA_seed91003', 'minus_C_IID_S-DFA_seed91004', 'minus_C_IID_S-DFA_seed91005', 'minus_C_IID_S-DFA_seed91006', 'minus_C_IID_S-DFA_seed91007', 'minus_C_IID_S-DFA_seed91008', 'minus_C_IID_S-DFA_seed91009', 'minus_C_IID_S-DFA_seed91010', 'minus_U_IID_Benign_seed91001', 'minus_U_IID_Benign_seed91002', 'minus_U_IID_Benign_seed91003', 'minus_U_IID_Benign_seed91004', 'minus_U_IID_Benign_seed91005', 'minus_U_IID_Benign_seed91006', 'minus_U_IID_Benign_seed91007', 'minus_U_IID_Benign_seed91008', 'minus_U_IID_Benign_seed91009', 'minus_U_IID_Benign_seed91010', 'minus_U_IID_F Flip_seed91001', 'minus_U_IID_F Flip_seed91002', 'minus_U_IID_F Flip_seed91003', 'minus_U_IID_F Flip_seed91004', 'minus_U_IID_F Flip_seed91005', 'minus_U_IID_F Flip_seed91006', 'minus_U_IID_F Flip_seed91007', 'minus_U_IID_F Flip_seed91008', 'minus_U_IID_F Flip_seed91009', 'minus_U_IID_F Flip_seed91010', 'minus_U_IID_FedSA_seed91001', 'minus_U_IID_FedSA_seed91002', 'minus_U_IID_FedSA_seed91003', 'minus_U_IID_FedSA_seed91004', 'minus_U_IID_FedSA_seed91005', 'minus_U_IID_FedSA_seed91006', 'minus_U_IID_FedSA_seed91007', 'minus_U_IID_FedSA_seed91008', 'minus_U_IID_FedSA_seed91009', 'minus_U_IID_FedSA_seed91010', 'minus_U_IID_S-DFA_seed91001', 'minus_U_IID_S-DFA_seed91002', 'minus_U_IID_S-DFA_seed91003', 'minus_U_IID_S-DFA_seed91004', 'minus_U_IID_S-DFA_seed91005', 'minus_U_IID_S-DFA_seed91006', 'minus_U_IID_S-DFA_seed91007', 'minus_U_IID_S-DFA_seed91008', 'minus_U_IID_S-DFA_seed91009', 'minus_U_IID_S-DFA_seed91010', 'minus_U_IID_Sp-DFA_seed91001', 'minus_U_IID_Sp-DFA_seed91002', 'minus_U_IID_Sp-DFA_seed91003', 'minus_U_IID_Sp-DFA_seed91004', 'minus_U_IID_Sp-DFA_seed91005', 'minus_U_IID_Sp-DFA_seed91006', 'minus_U_IID_Sp-DFA_seed91007', 'minus_U_IID_Sp-DFA_seed91008', 'minus_U_IID_Sp-DFA_seed91009', 'minus_U_IID_Sp-DFA_seed91010', 'minus_U_non-IID_Benign_seed91001', 'minus_U_non-IID_Benign_seed91002', 'minus_U_non-IID_Benign_seed91003', 'minus_U_non-IID_Benign_seed91004', 'minus_U_non-IID_Benign_seed91005', 'minus_U_non-IID_Benign_seed91006', 'minus_U_non-IID_Benign_seed91007', 'minus_U_non-IID_Benign_seed91008', 'minus_U_non-IID_Benign_seed91009', 'minus_U_non-IID_Benign_seed91010', 'minus_U_non-IID_F Flip_seed91001', 'minus_U_non-IID_F Flip_seed91002', 'minus_U_non-IID_F Flip_seed91003', 'minus_U_non-IID_F Flip_seed91004', 'minus_U_non-IID_F Flip_seed91005', 'minus_U_non-IID_F Flip_seed91006', 'minus_U_non-IID_F Flip_seed91007', 'minus_U_non-IID_F Flip_seed91008', 'minus_U_non-IID_F Flip_seed91009', 'minus_U_non-IID_F Flip_seed91010', 'minus_U_non-IID_FedSA_seed91001', 'minus_U_non-IID_FedSA_seed91002', 'minus_U_non-IID_FedSA_seed91003', 'minus_U_non-IID_FedSA_seed91004', 'minus_U_non-IID_FedSA_seed91005', 'minus_U_non-IID_FedSA_seed91006', 'minus_U_non-IID_FedSA_seed91007', 'minus_U_non-IID_FedSA_seed91008', 'minus_U_non-IID_FedSA_seed91009', 'minus_U_non-IID_FedSA_seed91010', 'minus_U_non-IID_S-DFA_seed91001', 'minus_U_non-IID_S-DFA_seed91002', 'minus_U_non-IID_S-DFA_seed91003', 'minus_U_non-IID_S-DFA_seed91004', 'minus_U_non-IID_S-DFA_seed91005', 'minus_U_non-IID_S-DFA_seed91006', 'minus_U_non-IID_S-DFA_seed91007', 'minus_U_non-IID_S-DFA_seed91008', 'minus_U_non-IID_S-DFA_seed91009', 'minus_U_non-IID_S-DFA_seed91010', 'minus_U_non-IID_Sp-DFA_seed91001', 'minus_U_non-IID_Sp-DFA_seed91002', 'minus_U_non-IID_Sp-DFA_seed91003', 'minus_U_non-IID_Sp-DFA_seed91004', 'minus_U_non-IID_Sp-DFA_seed91005', 'minus_U_non-IID_Sp-DFA_seed91006', 'minus_U_non-IID_Sp-DFA_seed91007', 'minus_U_non-IID_Sp-DFA_seed91008', 'minus_U_non-IID_Sp-DFA_seed91009', 'minus_U_non-IID_Sp-DFA_seed91010']
EXCLUDED_PRIOR_IDS = ['minus_C_IID_Benign_seed91001', 'minus_C_IID_Benign_seed91002', 'minus_C_IID_Benign_seed91003', 'minus_C_IID_Benign_seed91004', 'minus_C_IID_Benign_seed91005', 'minus_C_IID_Benign_seed91006', 'minus_C_IID_Benign_seed91007', 'minus_C_IID_Benign_seed91008', 'minus_C_IID_Benign_seed91009', 'minus_C_IID_Benign_seed91010', 'minus_C_IID_F Flip_seed91001', 'minus_C_IID_F Flip_seed91002', 'minus_C_IID_F Flip_seed91003', 'minus_C_IID_F Flip_seed91004', 'minus_C_IID_F Flip_seed91005', 'minus_C_IID_F Flip_seed91006', 'minus_C_IID_F Flip_seed91007', 'minus_C_IID_F Flip_seed91008', 'minus_C_IID_F Flip_seed91009', 'minus_C_IID_F Flip_seed91010', 'minus_C_IID_FedSA_seed91001', 'minus_C_IID_FedSA_seed91002', 'minus_C_IID_FedSA_seed91003', 'minus_C_IID_FedSA_seed91004', 'minus_C_IID_FedSA_seed91005', 'minus_C_IID_FedSA_seed91006', 'minus_C_IID_FedSA_seed91007', 'minus_C_IID_FedSA_seed91008', 'minus_C_IID_FedSA_seed91009', 'minus_C_IID_FedSA_seed91010', 'minus_C_IID_S-DFA_seed91001', 'minus_C_IID_S-DFA_seed91002', 'minus_C_IID_S-DFA_seed91003', 'minus_C_IID_S-DFA_seed91004', 'minus_C_IID_S-DFA_seed91005', 'minus_C_IID_S-DFA_seed91006', 'minus_U_IID_Benign_seed91001', 'minus_U_IID_Benign_seed91002', 'minus_U_IID_Benign_seed91003', 'minus_U_IID_Benign_seed91004', 'minus_U_IID_Benign_seed91005', 'minus_U_IID_Benign_seed91006', 'minus_U_IID_Benign_seed91007', 'minus_U_IID_Benign_seed91008', 'minus_U_IID_Benign_seed91009', 'minus_U_IID_Benign_seed91010', 'minus_U_IID_F Flip_seed91001', 'minus_U_IID_F Flip_seed91002', 'minus_U_IID_F Flip_seed91003', 'minus_U_IID_F Flip_seed91004', 'minus_U_IID_F Flip_seed91005', 'minus_U_IID_F Flip_seed91006', 'minus_U_IID_F Flip_seed91007', 'minus_U_IID_F Flip_seed91008', 'minus_U_IID_F Flip_seed91009', 'minus_U_IID_F Flip_seed91010', 'minus_U_IID_FedSA_seed91001', 'minus_U_IID_FedSA_seed91002', 'minus_U_IID_FedSA_seed91003', 'minus_U_IID_FedSA_seed91004', 'minus_U_IID_FedSA_seed91005', 'minus_U_IID_FedSA_seed91006', 'minus_U_IID_FedSA_seed91007', 'minus_U_IID_FedSA_seed91008', 'minus_U_IID_FedSA_seed91009', 'minus_U_IID_FedSA_seed91010', 'minus_U_IID_S-DFA_seed91001', 'minus_U_IID_S-DFA_seed91002', 'minus_U_IID_S-DFA_seed91003', 'minus_U_IID_S-DFA_seed91004', 'minus_U_IID_S-DFA_seed91005', 'minus_U_IID_S-DFA_seed91006', 'minus_U_IID_S-DFA_seed91007', 'minus_U_IID_S-DFA_seed91008', 'minus_U_IID_S-DFA_seed91009', 'minus_U_IID_S-DFA_seed91010', 'minus_U_IID_Sp-DFA_seed91001', 'minus_U_IID_Sp-DFA_seed91002', 'minus_U_IID_Sp-DFA_seed91003', 'minus_U_IID_Sp-DFA_seed91004', 'minus_U_IID_Sp-DFA_seed91005', 'minus_U_IID_Sp-DFA_seed91006', 'minus_U_IID_Sp-DFA_seed91007', 'minus_U_IID_Sp-DFA_seed91008', 'minus_U_IID_Sp-DFA_seed91009', 'minus_U_IID_Sp-DFA_seed91010', 'minus_U_non-IID_Benign_seed91001', 'minus_U_non-IID_Benign_seed91002', 'minus_U_non-IID_Benign_seed91003', 'minus_U_non-IID_Benign_seed91004', 'minus_U_non-IID_Benign_seed91005', 'minus_U_non-IID_Benign_seed91006', 'minus_U_non-IID_Benign_seed91007', 'minus_U_non-IID_Benign_seed91008', 'minus_U_non-IID_Benign_seed91009', 'minus_U_non-IID_Benign_seed91010', 'minus_U_non-IID_F Flip_seed91001', 'minus_U_non-IID_F Flip_seed91002', 'minus_U_non-IID_F Flip_seed91003', 'minus_U_non-IID_F Flip_seed91004', 'minus_U_non-IID_F Flip_seed91005', 'minus_U_non-IID_F Flip_seed91006', 'minus_U_non-IID_F Flip_seed91007', 'minus_U_non-IID_F Flip_seed91008', 'minus_U_non-IID_F Flip_seed91009', 'minus_U_non-IID_F Flip_seed91010', 'minus_U_non-IID_FedSA_seed91001', 'minus_U_non-IID_FedSA_seed91002', 'minus_U_non-IID_FedSA_seed91003', 'minus_U_non-IID_FedSA_seed91004', 'minus_U_non-IID_FedSA_seed91005', 'minus_U_non-IID_FedSA_seed91006', 'minus_U_non-IID_FedSA_seed91007', 'minus_U_non-IID_FedSA_seed91008', 'minus_U_non-IID_FedSA_seed91009', 'minus_U_non-IID_FedSA_seed91010', 'minus_U_non-IID_S-DFA_seed91001', 'minus_U_non-IID_S-DFA_seed91002', 'minus_U_non-IID_S-DFA_seed91003', 'minus_U_non-IID_S-DFA_seed91004', 'minus_U_non-IID_S-DFA_seed91005', 'minus_U_non-IID_S-DFA_seed91006', 'minus_U_non-IID_S-DFA_seed91007', 'minus_U_non-IID_S-DFA_seed91008', 'minus_U_non-IID_S-DFA_seed91009', 'minus_U_non-IID_S-DFA_seed91010', 'minus_U_non-IID_Sp-DFA_seed91001', 'minus_U_non-IID_Sp-DFA_seed91002', 'minus_U_non-IID_Sp-DFA_seed91003', 'minus_U_non-IID_Sp-DFA_seed91004', 'minus_U_non-IID_Sp-DFA_seed91005', 'minus_U_non-IID_Sp-DFA_seed91006', 'minus_U_non-IID_Sp-DFA_seed91007', 'minus_U_non-IID_Sp-DFA_seed91008', 'minus_U_non-IID_Sp-DFA_seed91009', 'minus_U_non-IID_Sp-DFA_seed91010']
REPLAY_IDS = ['minus_C_IID_S-DFA_seed91007', 'minus_C_IID_S-DFA_seed91008', 'minus_C_IID_S-DFA_seed91009', 'minus_C_IID_S-DFA_seed91010']
TOLERANCE = 1e-12
VIEWS = ['native', 'raw', 'shared_calibration']
BASELINE_INVENTORY_SHA = '3486a9f185f40294553de9e487e9d9b5d142098a819f6cfb3adbbd2d6a6420cd'
V2_SHA = '8476d651bd281d97d6fed42feac5569b45ab0e881b047a7962201e6c0b300803'
V3_SHA = 'abb4560dd1c6752b9017a739381d2c54e68f52e278075c8e225a1c67df35cc6e'
EVALUATOR_SHA = '805eedf1fb08137cd86a543a80c83b9527e5c8937be02f7d2dca83a33b86e04c'
EVIDENCE_V4_SHA = '3d78db7e67275dce2d52bc8927dd86fc1c517708224a994345b70cadc56427ef'
MANIFEST_SHA = '498ed0e033ef6eb5532286820ec987af3b44ca6251d8a84c42ce9c3e5ff5ab2d'
INSPECTION_SHA = '426397ba48fbc679bcb88b579d53a05487a94ee495565af59a7de1145e87e515'
CORE_SHA = 'cdd5586517d6a5d51a455d01384dd3093f5bd79947f6ba829bfab42520cb5eed'
TRAIN_IDS_SHA = '46d42484d5b5f53af8747fcf44ee11a0b051af27f10383d32254faee0311bc99'
VALID_IDS_SHA = '64a15cf28caf1d177ac3dcf96a4408bc21091796974b243923ca947a37b554bf'
VARIANTS = ['Full', 'minus_U', 'minus_C', 'minus_A', 'minus_F', 'minus_V', 'minus_N', 'no_hard_screen', 'fixed_balanced']
ATTACKS = ['Benign', 'F Flip', 'FedSA', 'S-DFA', 'Sp-DFA']
DISTRIBUTIONS = {'IID': 5000.0, 'non-IID': 5.0}
IGNORE_RECIPE = {'seed', 'client_alpha', 'ablation_component', 'experiment_suite', 'experiment_tag', 'full_round_diagnostics'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for data in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def load(name, path, sha):
    require(digest(path) == sha, 'Sealed dependency changed: ' + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def cell(record):
    return record['distribution'], record['attack'], record['seed']


def full_reference(record):
    return {'id': record['id'], 'role': 'Full_reference_only', 'variant': 'Full',
            **{key: record[key] for key in ('method', 'distribution', 'attack', 'seed', 'actual_alpha', 'training_torch')},
            'baseline_inventory_sha256': BASELINE_INVENTORY_SHA,
            'baseline_record_canonical_sha256': canonical(record),
            'checkpoint_sha256': record['checkpoint']['sha256'], 'result_sha256': record['result']['sha256'],
            'raw_job_sha256': record['raw_job']['sha256'], 'data_contract_canonical_sha256': canonical(record['data_contract']),
            'root_image_ids_sha256': record['data_contract']['root_image_ids_sha256'],
            'replay_required_here': False, 'weights_repacked_here': False,
            'views_status': 'REFERENCE_ONLY_REQUIRE_SHA_BOUND_BASELINE_REPLAY_ACCEPTANCE'}


VARIANT_COMPONENTS = {
    'minus_U': 'U', 'minus_C': 'C', 'minus_A': 'A', 'minus_F': 'F',
    'minus_V': 'V', 'minus_N': 'N', 'no_hard_screen': 'none', 'fixed_balanced': 'none',
}


def validate_variant_metadata(record):
    # Metadata does not prove an intervention: the original strict acceptor still
    # checks the frozen raw-job SHA, adapter SHA and all-round candidate ledger.
    variant, cfg = record['variant'], record['config']
    require(variant in VARIANT_COMPONENTS, 'Unknown variant; Full is reference-only')
    require(record['id'] == f"{variant}_{record['distribution']}_{record['attack']}_seed{record['seed']}",
            'Variant/scenario/seed ID mismatch')
    require(cfg['ablation_component'] == VARIANT_COMPONENTS[variant], 'Variant mask mismatch')
    require(cfg['experiment_suite'] == 'celeba_mechanism_v1' and cfg['experiment_tag'] == record['id']
            and cfg['full_round_diagnostics'] is True, 'Frozen mechanism bookkeeping changed')


def validate_inventory(inventory, baseline):
    """Accept actual accepted terminals and explicit Full references, never pending models."""
    require(inventory['scope'] == SCOPE and inventory['status'] == 'PREPARED_NOT_APPROVED', 'Wrong bridge scope/status')
    require(inventory['baseline_inventory_sha256'] == BASELINE_INVENTORY_SHA and inventory['mechanism_manifest_sha256'] == MANIFEST_SHA
            and inventory['mechanism_inspection_sha256'] == INSPECTION_SHA, 'Inventory authority changed')
    require(inventory['native_tolerance'] == TOLERANCE and inventory['views'] == VIEWS, 'Metric tolerance or view scope changed')
    full = {r['id']: r for r in baseline['records'] if r['method'] == 'GuardFed-AD2+'}
    expected_full = set(itertools.product(DISTRIBUTIONS, ATTACKS, range(91001, 91011)))
    require(len(full) == 100 and {cell(r) for r in full.values()} == expected_full, 'Incomplete/duplicate baseline Full cohort')
    refs = inventory['full_references']
    require(len(refs) == 100 and len({r['id'] for r in refs}) == 100 and {r['id'] for r in refs} == set(full), 'Duplicate/missing Full reference')
    require(all(r == full_reference(full[r['id']]) for r in refs), 'Full reference differs from accepted baseline model/data')
    records = inventory['records']
    require(len(records) == 140 and len({r['id'] for r in records}) == 140, 'This inventory contains exactly140 actual accepted terminals')
    require(set(r['id'] for r in records) == set(ACCEPTED_IDS), 'Actual accepted140 snapshot changed')
    require(inventory['excluded_prior_replay_ids'] == EXCLUDED_PRIOR_IDS and inventory['selected_replay_ids'] == REPLAY_IDS, 'Excluded-prior136/selected4 boundary changed')
    require(not set(full).intersection(r['id'] for r in records), 'Full may not enter a new replay cohort')
    require(all(r['variant'] in ('minus_U', 'minus_C') and r['distribution'] in DISTRIBUTIONS for r in records), 'Only adopted U100+C36 plus exact four C terminals; no other scope')
    paired = {cell(r): r for r in full.values()}
    for r in records:
        validate_variant_metadata(r)
        cfg, contract, control = r['config'], r['data_contract'], paired[cell(r)]
        require(r['method'] == r['source_method'] == 'GuardFed-AD2+' and r['variant'] in VARIANTS[1:], 'Mechanism must retain its actual method/variant')
        require((r['terminal_round'], r['original_split'], r['original_n_eval']) == (70, 'valid', 19867), 'Incomplete/non-valid terminal refused')
        require(canonical(cfg) == r['config_canonical_sha256'], 'Configuration identity changed')
        require(cfg['ablation_component'] == VARIANT_COMPONENTS[r['variant']] and cfg['seed'] == r['seed'] and cfg['client_alpha'] == r['actual_alpha'] == DISTRIBUTIONS[r['distribution']], 'Mask/seed/alpha changed')
        require(cfg['rounds'] == 70 and cfg['celeba_evaluation_split'] == 'valid' and not cfg['celeba_train_limit'] and not cfg['celeba_eval_limit'], 'Subset/test configuration refused')
        require(not cfg['synthetic_ratio'] and not cfg['root_label_noise'] and not cfg['root_sensitive_noise'] and cfg['ad2_calibration_enabled'] is True, 'Original clean root/native calibration required')
        require({k: v for k, v in cfg.items() if k not in IGNORE_RECIPE} == {k: v for k, v in control['config'].items() if k not in IGNORE_RECIPE}, 'Non-intervention recipe differs from paired Full')
        require(contract == control['data_contract'] and r['paired_full'] == full_reference(control), 'Paired Full root/train/valid/cache/model support differs')
        require(contract['train_image_ids_sha256'] == TRAIN_IDS_SHA and contract['evaluation_image_ids_sha256'] == VALID_IDS_SHA, 'Train/valid order differs')
        require(len(contract['client_sample_counts']) == 20 and sum(contract['client_sample_counts']) == 146493, 'Root/client support differs')
        require(r['source_hashes']['scripts/reproduce_paper_tables.py'] == CORE_SHA and r['training_torch'] == '2.11.0+cu128', 'Core/environment differs')
        require(r['adapter_source_hashes'] == inventory['mechanism_adapter_hashes'], 'Mechanism adapter lineage changed')
        require(r['source_hashes'] == inventory['mechanism_source_hashes'] and r['protocol_sha256'] == inventory['mechanism_protocol_sha256'], 'Frozen source/protocol changed')
        require(all(type(r['prior_validation_metrics'][k]) in (float, int) and math.isfinite(r['prior_validation_metrics'][k]) for k in ('accuracy', 'aeod', 'aspd')), 'Nonfinite/undefined terminal metrics')
        for kind, filename in [('checkpoint', 'model.pt'), ('result', 'result.json'), ('raw_job', r['id'] + '.json')]:
            a = r[kind]
            member = PurePosixPath(a['member'])
            require(not member.is_absolute() and '..' not in member.parts and member.name == filename and a['bytes'] > 0, 'Unsafe/mixed artifact member')
            target = r['runtime_paths'][kind]
            expected = r['original_job'] if kind == 'raw_job' else str(PurePosixPath(r['original_remote_output']) / filename)
            require(target == expected, 'Historical storage path changed')
        require(r['checkpoint']['sha256'] == r['accepted_v4_row']['checkpoint_sha256'], 'Mixed terminal checkpoint')
        require(r['manifest_entry']['id'] == r['accepted_v4_row']['id'] == r['id']
                and r['manifest_entry']['variant'] == r['accepted_v4_row']['variant'] == r['variant']
                and r['manifest_entry']['job'] == r['original_job']
                and r['manifest_entry']['output'] == r['accepted_v4_row']['output'] == r['original_remote_output'], 'Mixed manifest/acceptance record')
        files = r['accepted_v4_row']['files']
        require(all(files.get(r['runtime_paths'][k]) == r[k]['sha256'] for k in ('checkpoint', 'result', 'raw_job')), 'Original accepted file identity changed')
        require(r['raw_job']['sha256'] == r['manifest_entry']['job_sha256'], 'Manifest job bytes differ')
    missing = inventory['pending_new_ids_no_checkpoint']
    require(len(missing) == len(set(missing)) == 660 and not set(missing).intersection(r['id'] for r in records), 'Invalid pending cohort')
    expected_ids = {f'{v}_{d}_{a}_seed{s}' for v, d, a, s in itertools.product(VARIANTS[1:], DISTRIBUTIONS, ATTACKS, range(91001, 91011))}
    require(set(missing) | {r['id'] for r in records} == expected_ids, 'Missing/foreign planned mechanism ID')
    require(inventory['new_image_inference_performed'] is False and inventory['full_weights_repacked'] == 0, 'Preparation cannot claim inference or copy Full weights')
    return records


def require_approval(approval, inventory_sha, selected_id, bridge_sha):
    require(approval.get('status') == 'APPROVED_BOUNDED_MECHANISM_VALID_REPLAY_ONLY' and approval.get('scope') == SCOPE,
            'No bounded mechanism valid replay approval')
    require(approval.get('inventory_sha256') == inventory_sha and approval.get('bridge_sha256') == bridge_sha, 'Approval binds another inventory/source')
    ids = approval.get('selected_ids', [])
    require(ids == REPLAY_IDS and len(ids) == len(set(ids)) == 4 and selected_id in ids, 'Approval contains pending/Full/foreign/duplicate IDs')
    require(approval.get('device') == 'cpu' and approval.get('compute_threads') == 8 and approval.get('max_processes') == 1, 'Only one eight-thread CPU worker can be prepared here')
    cpus = approval.get('allowed_cpus', [])
    require(cpus == list(range(112, 120)), 'Explicit8 CPU budget required')
    require(approval.get('target_split') == 'valid' and approval.get('final_test_dispatch') is False and approval.get('native_tolerance') == TOLERANCE,
            'Approval changed split/tolerance/final scope')


def bind_runtime(inventory_path, inventory_sha, dependency_paths, repo, model_id, approval_path, approval_sha):
    """Future authorized worker interface. Creating it loads no image/prediction.

    Current preparation never calls this function. No runner/service is created.
    The existing foreground coordinator can call returned replay/accept functions
    after root reviews code, resource budget and external approval SHA.
    """
    require(sys.platform == 'linux' and not sys.flags.optimize, 'Unchanged Linux runtime without -O required')
    require(digest(inventory_path) == inventory_sha and digest(approval_path) == approval_sha, 'Inventory/approval bytes changed')
    inventory, approval = read(inventory_path), read(approval_path)
    require_approval(approval, inventory_sha, model_id, digest(__file__))
    require(set(os.sched_getaffinity(0)) == set(approval['allowed_cpus']), 'Worker is not on its separately approved CPUs')
    d = {k: Path(v) for k, v in dependency_paths.items()}
    require(digest(d['baseline_inventory']) == BASELINE_INVENTORY_SHA, 'Baseline inventory changed')
    baseline = read(d['baseline_inventory'])
    records = validate_inventory(inventory, baseline)
    record = next(r for r in records if r['id'] == model_id)
    require(digest(d['v2']) == V2_SHA and digest(d['evaluator']) == EVALUATOR_SHA, 'Scientific replay/evaluator changed')
    v3 = load('mechanism_bridge_sealed_v3', d['v3'], V3_SHA)
    v2 = v3.v2
    require(digest(v2.__file__) == V2_SHA and Path(v2.__file__).resolve() == d['v2'].resolve(), 'Sealed v3 imported a foreign replay module')
    require(v2.TOLERANCE == TOLERANCE and v2.VIEWS == VIEWS and v2.torch.__version__ == '2.11.0+cu128', 'Runtime/scientific tolerance changed')
    for task in Path('/proc/self/task').glob('*'):
        try:
            os.sched_setaffinity(int(task.name), approval['allowed_cpus'])
        except ProcessLookupError:
            pass
    require(digest(d['manifest']) == MANIFEST_SHA and digest(d['protocol']) == inventory['mechanism_protocol_sha256'], 'Frozen mechanism manifest/protocol changed')
    evidence = load('mechanism_bridge_evidence_v4', d['evidence_v4'], EVIDENCE_V4_SHA)
    manifest = read(d['manifest'])
    adapter_dir = Path(next(iter(manifest['adapter_hashes']))).parent
    adapter = adapter_dir / 'adapter.py'
    if 'adapter' in sys.modules:
        require(digest(sys.modules['adapter'].__file__) == manifest['adapter_hashes'][str(adapter)], 'Foreign cached adapter module')
    original, worker = evidence.validators(repo, adapter_dir, manifest)
    evaluator = load('mechanism_bridge_sealed_evaluator', d['evaluator'], EVALUATOR_SHA)
    core = load('mechanism_bridge_frozen_core', repo / 'scripts/reproduce_paper_tables.py', CORE_SHA)
    cnn = load('mechanism_bridge_original_cnn', repo / 'src/celeba_data.py', record['source_hashes']['src/celeba_data.py'])
    full = next(r for r in baseline['records'] if r['id'] == record['paired_full']['id'])
    paths = {k: Path(v) for k, v in record['runtime_paths'].items()}
    sources = {Path(__file__): digest(__file__), Path(inventory_path): inventory_sha, Path(approval_path): approval_sha,
               d['v2']: V2_SHA, d['v3']: V3_SHA, d['evaluator']: EVALUATOR_SHA, d['evidence_v4']: EVIDENCE_V4_SHA,
               d['baseline_inventory']: BASELINE_INVENTORY_SHA, d['manifest']: MANIFEST_SHA,
               d['protocol']: inventory['mechanism_protocol_sha256']}
    sources.update({v2.inside(repo, rel): sha for rel, sha in record['source_hashes'].items()})
    sources.update({Path(name): sha for name, sha in record['adapter_source_hashes'].items()})
    artifacts = {Path(name): sha for name, sha in record['accepted_v4_row']['files'].items()}
    before_sources, before_artifacts = v3.full_hashes(sources), v3.full_hashes(artifacts)

    def validate(_original, actual_record, actual_repo):
        require(actual_record == record and actual_repo == repo, 'Bound record/repository changed')
        require(v3.full_hashes(artifacts) == before_artifacts, 'Original model/result/job/audit changed')
        job = read(paths['raw_job'])
        result = evidence.accept_new(original, worker, job, record['manifest_entry'], manifest, full)
        require(result is not None, 'Partial/unaccepted terminal cannot be replayed')
        require(job['config'] == record['config'] and job['source_hashes'] == record['source_hashes']
                and job['variant'] == record['variant'] and job['output'] == record['original_remote_output'], 'Original identity differs from inventory')
        require(result['metrics'] == record['prior_validation_metrics'] and result['data_contract']['image_data_contract'] == record['data_contract'], 'Original terminal metrics/data changed')
        return result

    by_member = {record[k]['member']: paths[k] for k in ('checkpoint', 'result', 'raw_job')}
    def located(_repo, relative):
        return by_member[str(relative)] if str(relative) in by_member else v2.inside(_repo, relative)
    inference = v3.private(v2.replay_one, inside=located, validate_original=validate)

    def replay(output, wall_seconds=1800):
        output = Path(output)
        failure_path = output.with_name(output.name + '.bridge_failure.json')
        require(str(output) == approval['output'] and not output.exists() and not failure_path.exists()
                and not output.resolve().is_relative_to(repo.resolve()), 'Only approved new isolated output allowed; preserve partial/failure')
        require(1 <= wall_seconds <= 3600, 'Bounded wall time required')
        proof = {'scope': SCOPE, 'status': 'FAILED', 'id': model_id, 'variant': record['variant'],
                 'inventory_sha256': inventory_sha, 'inventory_record_sha256': canonical(record),
                 'bridge_source_sha256': digest(__file__), 'v2_source_sha256': V2_SHA, 'v4_source_sha256': EVIDENCE_V4_SHA,
                 'approval_sha256': approval_sha, 'source_before': before_sources, 'artifact_before': before_artifacts,
                 'paired_full_reference': record['paired_full'],
                 'views': VIEWS, 'native_tolerance': TOLERANCE, 'no_new_training': True,
                 'test_images_inferred': False, 'test_labels_accessed_by_replay': False,
                 'original_training_loader_materialized_all_split_metadata': True,
                 'final_protocol_frozen': False, 'all900_mechanism_replayed': False,
                 'claim_limit': 'One accepted terminal valid-only implementation replay; Full is referenced, not replayed here.'}
        try:
            v3.check_source_tokens(before_sources)
            ids, y, s, metadata_receipt = v2.metadata(repo)
            inference(core, original, cnn, evaluator, record, repo, ids, y, s, output, wall_seconds)
            proof.update(source_after=v3.full_hashes(sources), artifact_after=v3.full_hashes(artifacts), metadata_read_receipt=metadata_receipt)
            require(proof['source_after'] == before_sources and proof['artifact_after'] == before_artifacts, 'Inputs changed during replay')
            proof.update(status='MECHANISM_NATIVE_VALID_REPLAY_PASS', scientific_body_receipt_sha256=digest(output / 'receipt.json'))
            save_new(output / 'bridge_receipt.json', proof)
            return proof
        except BaseException as exc:
            import traceback
            proof.update(status='FAILED', error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
            for name, pins in [('source_after', sources), ('artifact_after', artifacts)]:
                try:
                    proof[name] = v3.full_hashes(pins)
                except Exception as identity_error:
                    proof[name + '_error'] = str(identity_error)
            save_new(failure_path, proof)
            raise

    def accept(output):
        """Reuses sealed scoring/fit/root/native logic; no image or model inference."""
        output = Path(output)
        require(str(output) == approval['output'], 'Wrong approved output')
        p, r = read(output / 'bridge_receipt.json'), read(output / 'receipt.json')
        require(p['scope'] == SCOPE and p['status'] == 'MECHANISM_NATIVE_VALID_REPLAY_PASS' and p['id'] == r['id'] == model_id, 'Incomplete/foreign replay receipt')
        require(p['variant'] == record['variant'] and p['paired_full_reference'] == record['paired_full']
                and p['views'] == VIEWS and p['native_tolerance'] == TOLERANCE, 'Replay variant/pair/view identity changed')
        require(all(r[k] == record[k] for k in ('method', 'distribution', 'attack', 'seed', 'config_canonical_sha256'))
                and r['original_training_torch'] == record['training_torch'] and r['valid_n'] == 19867
                and r['valid_image_ids_sha256'] == VALID_IDS_SHA, 'Replay config/environment/split metadata changed')
        require(p['inventory_sha256'] == inventory_sha and p['inventory_record_sha256'] == r['model_inventory_record_sha256'] == canonical(record), 'Mixed replay inventory')
        require(p['bridge_source_sha256'] == digest(__file__) and p['v2_source_sha256'] == V2_SHA and p['v4_source_sha256'] == EVIDENCE_V4_SHA
                and p['approval_sha256'] == approval_sha and p['scientific_body_receipt_sha256'] == digest(output / 'receipt.json'), 'Changed receipt/source approval')
        require(p['source_before'] == p['source_after'] == v3.full_hashes(sources)
                and p['artifact_before'] == p['artifact_after'] == v3.full_hashes(artifacts), 'Input identity changed after replay')
        require(r['status'] == 'NATIVE_VALID_REPLAY_PASS' and r['runtime']['device'] == 'cpu' and r['runtime']['cuda_device_count'] == 0
                and not r['optimizer_created'] and not r['gradients_created'] and not r['test_labels_accessed'], 'Inference-only contract failed')
        require(r['checkpoint_sha256'] == record['checkpoint']['sha256'] and r['original_result_sha256'] == record['result']['sha256']
                and r['original_job_sha256'] == record['raw_job']['sha256'], 'Mixed terminal/checkpoint artifacts')
        for resources in (r['before_resources'], r['after_resources']):
            require(all(set(cpus) == set(approval['allowed_cpus']) for cpus in resources['thread_cpu_affinities'].values()), 'Replay escaped approved CPUs')
        result = validate(original, record, repo)
        ids, y, s, metadata_receipt = v2.metadata(repo)
        cfg = core.ExperimentConfig(**record['config'])
        root_ids, root_y, root_s, root_receipt = v2.rebuild_root(core, cfg, record, ids, y, s)
        require(root_receipt == r['root_reconstruction'] and metadata_receipt == p['metadata_read_receipt'] and r['weights_before'] == r['weights_after'], 'Root/metadata/weights changed')
        require(v2.digest(output / 'validation_predictions.npz') == r['prediction_arrays_sha256'], 'Saved predictions changed')
        with v2.np.load(output / 'validation_predictions.npz', allow_pickle=False) as z:
            require(v2.np.array_equal(z['root_image_ids'], root_ids) and v2.np.array_equal(z['valid_image_ids'], ids[162770:182637]), 'Prediction sample/order mismatch')
            fits = evaluator.fit_views(core, record['method'], z['root_margins'], root_y, root_s, cfg, VIEWS, evaluator.SHARED_CALIBRATION)
            require(canonical(fits) == canonical(r['fits']), 'Root-only fits changed')
            pred = evaluator.predict_views(z['valid_margins'], s[162770:], fits)
            require(all(v2.np.array_equal(pred[v], z['prediction_' + v]) for v in VIEWS), 'Predictions contradict fixed margins/ties/thresholds')
            scored = evaluator.evaluate_frozen_predictions(pred, y[162770:], s[162770:])
            comparison = v2.check_native(scored['native'], result['metrics'])
            require(scored == r['views'] and comparison == r['native_comparison'] and comparison['accepted'], 'Common-checkpoint metrics/native1e-12 failed')
        require(v3.full_hashes(sources) == p['source_before'] and v3.full_hashes(artifacts) == p['artifact_before'], 'Acceptance inputs changed')
        return {'scope': SCOPE, 'status': 'MECHANISM_VALID_THREE_VIEWS_ACCEPTED', 'id': model_id, 'variant': record['variant'],
                'views': scored, 'native_comparison': comparison, 'checkpoint_sha256': record['checkpoint']['sha256'],
                'bridge_receipt_sha256': digest(output / 'bridge_receipt.json'), 'inventory_sha256': inventory_sha,
                'paired_full_reference': record['paired_full'], 'new_training': False, 'test_inference': False,
                'final_protocol_status': 'PREPARED_NOT_FROZEN'}
    return {'record': copy.deepcopy(record), 'validate_original': validate, 'replay_one': replay, 'accept_saved_predictions': accept}


def reference_baseline_full(inventory, baseline, acceptance_path, acceptance_sha):
    """Join accepted baseline Full identities only; never infer or copy weights."""
    validate_inventory(inventory, baseline)
    require(digest(acceptance_path) == acceptance_sha, 'Baseline acceptance SHA is not externally bound')
    acceptance = read(acceptance_path)
    require(acceptance['scope'] == 'VALID_ONLY_IMPLEMENTATION_REPLAY_V3' and acceptance['inventory_sha256'] == BASELINE_INVENTORY_SHA,
            'Baseline acceptance is from another cohort')
    require(acceptance.get('status') == 'SELECTED_VALID_REPLAY_ACCEPTED', 'Provide a strict individual baseline batch acceptance; cumulative labels alone are insufficient')
    require(acceptance.get('valid_image_ids_sha256') == VALID_IDS_SHA and acceptance.get('calibration_core_sha256') == CORE_SHA
            and acceptance.get('v2_source_sha256') == V2_SHA, 'Baseline native/raw/shared scientific source/split differs')
    ids = acceptance['accepted_ids']
    require(len(ids) == len(set(ids)) == acceptance['accepted_n'] and set(ids) <= {r['id'] for r in baseline['records']}, 'Duplicate/foreign baseline acceptance IDs')
    require(acceptance.get('max_abs_native_metric_difference') is not None and 0 <= acceptance['max_abs_native_metric_difference'] <= TOLERANCE
            and acceptance.get('test_labels_accessed') is False, 'Native baseline tolerance/scope failed')
    if 'invalid' in acceptance:
        require(not acceptance['invalid'], 'Invalid baseline records require review')
    accepted_full = [r for r in inventory['full_references'] if r['id'] in ids]
    return {'scope': SCOPE, 'accepted_full_reference_count': len(accepted_full), 'reference_only': True,
            'baseline_acceptance_sha256': acceptance_sha, 'full_references': accepted_full,
            'missing_full_replay_ids': sorted({r['id'] for r in inventory['full_references']} - set(ids)),
            'new_full_inference': 0, 'full_weights_repacked': 0}
