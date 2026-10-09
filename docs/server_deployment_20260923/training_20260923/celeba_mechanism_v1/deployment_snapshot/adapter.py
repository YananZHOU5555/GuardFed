"""Isolated mechanism controls over the immutable, accepted GuardFed core."""
from contextlib import contextmanager
import inspect

VARIANTS = ('Full', 'minus_U', 'minus_C', 'minus_A', 'minus_F', 'minus_V',
            'minus_N', 'no_hard_screen', 'fixed_balanced')


def without_hard_screen(core, original):
    # Both the distance/alignment gate and top-k truncation belong to this control.
    source = inspect.getsource(original)
    start = source.index('    dist_med, dist_scale = robust_median_mad(distances)\n')
    end = source.index('    selected_norms = ', start)
    removed = source[start:end]
    assert removed.count('hard_gate = ') == 2
    assert removed.count('select_top_scores(') == 1
    source = source[:start] + ('    hard_gate = list(range(n))\n'
                              '    selected = list(range(n))\n\n') + source[end:]
    namespace = dict(core.__dict__)
    exec(compile(source, '<GuardFed-no-hard-screen>', 'exec'), namespace)
    return namespace[original.__name__]


@contextmanager
def mechanism(core, variant):
    """Patch only an isolated worker module; preserve the original source bytes."""
    if variant not in VARIANTS:
        raise ValueError(variant)
    original = core.guardfed_act_aggregate
    selector = core.guardfed_ad2plus_adaptive_aggregate
    ledger = []
    if variant == 'Full':
        yield ledger
        return
    aggregate = without_hard_screen(core, original) if variant == 'no_hard_screen' else original

    def checked(*args, **kwargs):
        cfg = args[3]
        expected = variant[-1] if variant.startswith('minus_') else 'none'
        assert cfg.ablation_component == expected
        update, info = aggregate(*args, **kwargs)
        if expected in 'UCAFV' and expected != 'none':
            assert all(t[expected] == 0.0 for t in info['component_contributions'])
        if expected == 'N':
            assert all(v == 1.0 for v in info['norm_clip_scales'])
        if variant == 'no_hard_screen':
            assert info['hard_gate_clients'] == info['selected_clients'] == list(range(len(args[0])))
        ledger.append({'variant': variant, 'component': expected,
                       'selected_count': len(info['selected_clients']),
                       'clients': len(args[0]), 'mask_verified': True})
        info['mechanism_variant'] = variant
        return update, info

    def fixed(updates, fairness, server_update, config, fairness_details=None,
              global_state=None, bundle=None, device=None):
        options = [dict(c) for c in core.ad2plus_candidate_overrides(config)
                   if c['candidate_name'] == 'balanced']
        assert len(options) == 1
        overrides = options[0]
        overrides.pop('candidate_name')
        fixed_config = core.clone_config_with(config, overrides)
        update, info = checked(updates, fairness, server_update, fixed_config,
                               fairness_details=fairness_details)
        info.update(ad2_plus_mode='fixed_balanced_without_candidate_selection',
                    fixed_candidate='balanced', fixed_candidate_config=overrides)
        return update, info

    core.guardfed_act_aggregate = checked
    if variant == 'fixed_balanced':
        core.guardfed_ad2plus_adaptive_aggregate = fixed
    try:
        yield ledger
    finally:
        core.guardfed_act_aggregate = original
        core.guardfed_ad2plus_adaptive_aggregate = selector


def verify_ledger(ledger, variant, rounds):
    expected = 0 if variant == 'Full' else rounds * (1 if variant == 'fixed_balanced' else 10)
    assert len(ledger) == expected, (variant, len(ledger), expected)
    assert all(x['mask_verified'] and x['variant'] == variant for x in ledger)
    return {'variant': variant, 'rounds': rounds, 'candidate_calls_verified': len(ledger),
            'expected_candidate_calls': expected, 'pass': True}
