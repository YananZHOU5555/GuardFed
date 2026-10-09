"""Pure job construction, called only by the externally approved binding step."""
import copy
import itertools


def definitions(protocol, adapter_hashes):
    candidate = protocol['selected_recipe']
    for distribution, attack, seed in itertools.product(protocol['distributions'], protocol['attacks'], protocol['seeds']):
        if seed == 91001 and attack in ('Benign', 'S-DFA'):
            continue
        yield make_job(protocol, adapter_hashes, distribution, attack, seed, 'fullcoverage')


def make_job(protocol, adapter_hashes, distribution, attack, seed, phase):
    candidate = protocol['selected_recipe']
    identity = f"{candidate['id']}_{distribution}_{attack}_seed{seed}_{phase}"
    cfg = dict(protocol['base_config'], rounds=70 if phase == 'fullcoverage' else 3,
               seed=seed, client_alpha=protocol['distributions'][distribution],
               learning_rate=candidate['learning_rate'], experiment_suite=protocol['version'], experiment_tag=identity)
    return dict(id=identity, dataset='celeba', method='FLGMM-author-code', distribution=distribution,
                attack=attack, phase=phase, evidence_stage='multi_seed_validation_coverage' if phase == 'fullcoverage' else 'pipeline_canary_only',
                config=cfg, adapter=copy.deepcopy(candidate['adapter']), tuning_candidate=candidate['id'],
                source_hashes=copy.deepcopy(protocol['source_hashes']), adapter_source_hashes=copy.deepcopy(adapter_hashes))
