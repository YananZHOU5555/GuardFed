# A50: five IID scenes, three-view candidate

The candidate contains all five IID scenarios for Full versus minus_A, with 10 matched model seeds per scenario. Only Sp-DFA adds new pairs to the accepted A40 table. All original 80 records (including serialized object bytes and order), 648 statistical scalars, and 324 display cells are unchanged. The adopted source contains one non-IID Benign seed; it is explicitly excluded from every displayed average.

Differences below are minus_A minus Full. ACC differences are percentage points; AEOD/ASPD use their original absolute gap scale. Positive ACC favors deletion, whereas negative gaps favor deletion. Uncertainty is the sample SD of paired differences across model seeds.

| Added Sp-DFA, 10 seeds | ΔACC (pp) | ΔAEOD | ΔASPD |
|---|---:|---:|---:|
| raw | -0.266 ± 0.612 | 0.00731 ± 0.02104 | 0.00145 ± 0.01756 |
| native | -0.230 ± 0.746 | -0.00950 ± 0.00897 | 0.00117 ± 0.01870 |
| shared_calibration | -0.230 ± 0.746 | -0.00950 ± 0.00897 | 0.00117 ± 0.01870 |

Deleting A reduces Sp-DFA accuracy in all 10/9/6 panels. Raw AEOD increases in all panels; native/shared AEOD decreases. Native/shared ASPD increases in all panels, while raw ASPD increases for 10/9 seeds but decreases for 6 seeds. These opposite directions are retained. This comparison does not show that A improves every fairness endpoint, and it does not isolate aggregation from native calibration.

The separate five-IID aggregate uses one within-seed mean over five scenes, followed by a mean and sample SD across seeds. It never treats 50 scenario-seed cells as 50 independent model seeds. Native/shared 10-seed paired differences are ΔACC −0.177 ± 0.454 pp, ΔAEOD −0.00048 ± 0.00533, ΔASPD −0.00190 ± 0.00941. Raw differences are −0.134 ± 0.400 pp, +0.00310 ± 0.00449, +0.00167 ± 0.00275. Thus the raw and calibrated aggregate tradeoffs also differ.

All three views use the same terminal round-70 checkpoint per record. AEOD here is the absolute TPR gap, not full equalized odds. Shared calibration is the original frozen root-only fit; this table builder performs no fit. Full replay devices are 3 CPU and 47 GPU, while all 50 minus_A replays use CPU; all 100 training records are cu128. Per-record source, driver, partition, checkpoint, calibration and receipt identities are retained. The broader Full100 source includes cu130 history, but that broader count is not this IID subset.

Seed91001 participated in selection. Validation was exposed during development; historical test exposure remains disclosed. The 9/6-seed subsets are predefined descriptive sensitivity panels, not untouched test confirmation. No endpoint is chosen here. No significance, universal necessity, causal-isolation, A100, full ten-scene coverage, or whole-rebuttal-completion claim follows.

Checks: 810 per-scene and 162 seed-first mean/SD scalars were recomputed using the original independent math.fsum/ddof1 functions; maximum difference 1.42e−14. The original grouped counts reproduce 900 metrics and satisfy 2400 count checks. Markdown and LaTeX contain all 486 formatted mean/SD cells. The LaTeX fragment is source-checked (18 balanced table environments), not compiled into a submission manuscript.

Two preparation failures are retained. The first was an ambiguous source replacement before any build source/output existed. The actual first build then rejected eight provenance path strings (backslash versus slash) on four old A40 records; no table output was created. The repaired reader only normalizes already SHA-checked predecessor path spellings; scientific functions and all source records are unchanged.
