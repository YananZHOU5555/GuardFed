# C100 complete coverage — author-review table candidate

The table contains Full100 and minus_C100: IID/non-IID × five attacks × ten shared seeds, all from round70 validation checkpoints. The new input is the separately root-adopted twenty-record chain (first1 + next19); no CNN, refitting, training or test evaluation was performed here. This completes this one ablation's coverage, not the other six controls or the whole revision.

All differences below are **minus_C − Full**. ACC differences are percentage points; AEOD/ASPD are absolute gaps. Positive ACC favors deletion, whereas positive gaps favor Full. AEOD means absolute TPR gap, not full equalized odds.

| New non-IID scene | View | n | ΔACC (pp) | ΔAEOD | ΔASPD |
|---|---|---:|---:|---:|---:|
| S-DFA | native/shared | 10 | +0.532038 | −0.00111223 | −0.00083292 |
| S-DFA | raw | 10 | +0.650828 | −0.01488832 | −0.00716885 |
| Sp-DFA | native/shared | 10 | +0.680525 | −0.00058488 | +0.00478524 |
| Sp-DFA | raw | 10 | +0.708210 | −0.00668878 | −0.00195855 |

The ten-seed S-DFA mean favors deletion on all three recorded metrics. Sp-DFA shows a native/shared accuracy–ASPD tradeoff. These results do not support a claim that every score component is indispensable. They are descriptive paired contrasts and do not identify pure aggregation causality or establish statistical significance. Raw/native/shared remain parallel views, without selecting a primary endpoint.

The n9/n6 panels are retained in full. The new scenes' ACC deltas remain positive in both subsets. For S-DFA native/shared, AEOD switches from −0.00111223 (n10) to +0.00000400 (n9) and +0.00315052 (n6). For Sp-DFA native/shared, AEOD switches to +0.00079077/+0.00033938; ASPD remains positive. Raw ASPD also changes sign in the n6 subset for both new scenes. No unfavorable subset has been dropped.

Balanced ten-scene summaries first average the ten scenes within each seed, then take the mean and sampleSD over the seed units. At n10, native/shared deletion deltas are +0.383903pp ACC, −0.00057975 AEOD, +0.00372478 ASPD; raw deltas are +0.422711pp, −0.00187609, +0.00086755. The positive calibrated ASPD difference persists in n9/n6. These are a supplementary balanced summary, not a replacement for the scene tables. The original five-IID seed-first summary is preserved byte-for-byte; the five-non-IID summary is separate.

Validation: 200 unique records, 100 pairs, 10 complete scenes; 1,620 scene mean/SD scalars, 810 displayed mean±SD cells, 1,800 metrics recomputed from group confusion counts, and 4,800 structural count checks. Independent fsum/sampleSD error is at most 1.43e−14. The two new seed-first summaries add 324 scalars. Original C80's 160 records, 1,296 statistics, 648 displayed cells and 162 IID aggregate scalars are unchanged.

Runtime evidence remains mixed: Full replay uses5 CPU and95 GPU checkpoints, with98 cu128 and2 cu130 training records. All100 minus_C replays use CPU and their training records report cu128. Each record retains the actual configuration, runtime and checkpoint identity. Seed91001's configuration-selection history, prior validation exposure and historical official-test exposure remain disclosed. These are not newly untouched test results.

`snapshot/TABLES.md` is the full paper-style table; `snapshot/tables.json` retains exact statistics; `snapshot/records.json` preserves all individual values and origins. `snapshot/SOURCE_BINDINGS.json` binds the actual ROOT_ADOPTION and200-index SHA plus each new scientific/strict receipt. Whole archive and model-member validation is inherited from those real root acceptances; this build reads only small accepted JSONs and does not repackage weights or arrays.

The output is a candidate awaiting independent root review and adoption. It has not modified the manuscript, canonical tables, STATE, Git or any training queue.
