"""Source-only helper proposed for insertion in a NEW bounded bridge; no runtime."""
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
