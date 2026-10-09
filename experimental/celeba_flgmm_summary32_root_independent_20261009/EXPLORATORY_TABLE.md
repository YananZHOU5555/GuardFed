# FLGMM 32-record exploratory recipe screen — independent verification

PASS: 32 unique original strict records, eight candidates, and four conditions per candidate (IID/non-IID × Benign/S-DFA), seed 91001, 70 rounds. All eight candidates and negative results are retained. Values below are arithmetic means across conditions; n=1 seed, no sample SD or significance.

| Tg | L | LR | ACC (%) ↑ | AEOD (%) ↓ | ASPD (%) ↓ | Frozen score | Scope |
|---:|---:|---:|---:|---:|---:|---:|:---|
| 10 | 2.0 | 0.0005 | 86.0409 | 5.1611 | 9.6136 | 0.83016075 | dominated |
| 10 | 2.0 | 0.001 | 87.8668 | 4.3409 | 10.0355 | 0.84847732 | Pareto |
| 10 | 3.0 | 0.0005 | 86.1617 | 5.1557 | 9.5941 | 0.83143417 | Pareto |
| 10 | 3.0 | 0.001 | 87.9310 | 4.6007 | 10.2838 | 0.84798355 | Pareto |
| 20 | 2.0 | 0.0005 | 86.0120 | 5.1500 | 9.6252 | 0.82985481 | Pareto |
| 20 | 2.0 | 0.001 | 87.8970 | 4.3895 | 10.0956 | 0.84852710 | score leader, Pareto |
| 20 | 3.0 | 0.0005 | 86.1089 | 5.2380 | 9.6562 | 0.83059470 | dominated |
| 20 | 3.0 | 0.001 | 87.9398 | 4.6889 | 10.3481 | 0.84774490 | ACC champion, Pareto |

Frozen-score leader: `FLGMM_Tg20_L2.0_lr0.001`; ACC champion: `FLGMM_Tg20_L3.0_lr0.001`. The three-metric Pareto set contains 6 candidates. Exact score ties: 0.

The score leader exceeds `FLGMM_Tg10_L2.0_lr0.001` by 4.97792046442e-05 score units (0.00497792046442 after multiplying score by 100). Both round to 84.85 on that latter scale at two decimals; this is a rounded near-tie, not an exact tie. Selection used unrounded scores and no epsilon; exact ties would use candidate lexical order.

The frozen score is calculated separately for each condition, then averaged over the four conditions: `ACC - .35*(.45*AEOD + .45*ASPD + .10*max(AEOD,ASPD)) - .10*max(0,max(AEOD,ASPD)-.06)`. Metrics in this expression are fractions. A score calculated from averaged metrics is not substituted.

This exposed valid-only search does not establish multi-seed paper performance, select an author endpoint, or adopt a formal-100 protocol/recipe. Author-code clustering differs from the paper introduction; upstream bounds behavior and the declared zero-standard-deviation extension remain disclosed in the frozen protocol. No new training, inference or final-test evaluation occurred in this audit.

The first SUMMARY attempt failed on a missing host-observation field in the exact legacy-two proof. Its failure record remains unchanged. The separate reader passed the exact two-record/21-source check and rejected nonlegacy missing-host proofs and source/checkpoint/length mutations. No absent host flag was fabricated.

SUMMARY32 SHA256: `d8f32441402f0ce7870bacc34e94de2a334133b2881d0a59ccf5123ef0781291`.
Final6 ROOT SHA256: `66097564f346b0dd5a0194ea7bd8c1413d51936819db086283f1d9a99a85738e`.
Reader SHA256: `303ee5a49e7f1ca3ee846b43e3e9f4bc70584a764e5c9bd459b06e730edc7985`.

All 32 condition scores and 32 candidate mean scalars match exactly. Original record identity, metrics, source/data pins and checkpoint hashes are preserved. The independent audit reads existing ROOT/offserver/archive hashes; it does not rerun strict acceptance or inspect tensors anew.
