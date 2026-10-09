# Source reuse and minimal changes

Production metric/statistic functions were not copied or modified. The native92
builder is the original generic render_mechanism_interim_tables_20261009.py; its
SHA remains023553f7ed63a5c4f152dbdaa7e02a9f9073e9530d5a66553a0ec6bf22628247.
Its evidence_v4 statistic/summarize SHA remains
3d78db7e67275dce2d52bc8927dd86fc1c517708224a994345b70cadc56427ef.

New build.py only locks actual181013Z104+100 input identities, requires exact
Full100/U100 ten-scene coverage while preservingC4 partial, compares all192 prior
records, calls the unchanged renderer, and exports original paired_per_seed plus
checkpoint identities and coverage metadata. Native92 output remains unmodified.

New verify.py adapts the oldnative92 independent fsum/sampleSD verification from
9 to10 scenes/486 to540 checks, compares every oldnine row exactly, checks100
original checkpoint pairs/300 delta scalars, and rejects four actual scope drifts.
It is verification code, not a new production scientific metric implementation.

The frozen ledger copy is byte-identical to actual8fc6b8b1... at snapshot time.
No future/current mutable ledger contents are silently adopted. The original full
inspection retains all204 records;C4 partial are not filtered out of source data.
